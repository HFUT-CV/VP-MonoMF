import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PoseDetector(nn.Module):
    """Detect camera pose: horizon and vanishing point"""
    def __init__(self, in_channels):
        super(PoseDetector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Horizon: represented as y = ax + b (2 parameters: a, b)
        self.horizon_head = nn.Conv2d(32, 2, 1)
        # Vanishing point: (xvp, yvp) - 2 parameters
        self.vanishing_point_head = nn.Conv2d(32, 2, 1)

    def forward(self, features):
        """
        Returns:
            horizon: (a, b) for y = ax + b
            vanishing_point: (xvp, yvp)
        """
        x = self.conv(features)
        horizon = self.horizon_head(x)  # (B, 2, H, W)
        vanishing_point = self.vanishing_point_head(x)  # (B, 2, H, W)
        
        # Average across spatial dimensions for global pose
        horizon = horizon.mean(dim=(-2, -1))  # (B, 2)
        vanishing_point = vanishing_point.mean(dim=(-2, -1))  # (B, 2)
        
        return horizon, vanishing_point


class ThreeD_Reconstruction(nn.Module):
    """3D Reconstruction (3DR) Module
    
    Corrects camera pose variance and reconstructs depth map based on camera projection.
    Uses camera transformation matrix to convert direct depth to global depth.
    """
    def __init__(self, in_channels):
        super(ThreeD_Reconstruction, self).__init__()
        self.pose_detector = PoseDetector(in_channels)

    def forward(self, depth_map, features, camera_intrinsics=None):
        """
        Args:
            depth_map: Zdir (B, 1, H, W) - direct depth from detector
            features: backbone features for pose detection
            camera_intrinsics: dict with 'focal_length', 'cx', 'cy'
        
        Returns:
            global_depth: Zglo (B, 1, H, W) - corrected global depth
            reliability: σglo (B, 1, H, W)
        """
        # Default camera intrinsics (KITTI dataset typical values)
        if camera_intrinsics is None:
            camera_intrinsics = {
                'focal_length': 721.5377,  # f (KITTI baseline)
                'cx': 609.5593,  # principal point x
                'cy': 172.8540,  # principal point y
            }

        # Step 1: Detect camera pose
        horizon, vanishing_point = self.pose_detector(features)
        
        # Step 2: Calculate rotation angles from horizon and vanishing point
        # Formula (6): θR = arctan(a), θp = arctan((xvp - cu) / f)
        a = horizon[:, 0]  # slope of horizon line
        b = horizon[:, 1]  # intercept of horizon line
        xvp = vanishing_point[:, 0]
        yvp = vanishing_point[:, 1]
        
        theta_R = torch.atan(a)  # roll angle
        f = camera_intrinsics['focal_length']
        cu = camera_intrinsics['cx']
        theta_P = torch.atan((xvp - cu) / (f + 1e-6))  # pitch angle
        
        # Step 3: Build transformation matrices
        # Formula (7), (8): A = AR @ AP
        batch_size = depth_map.size(0)
        A = self._build_transformation_matrix(theta_R, theta_P, batch_size)
        
        # Step 4: Convert depth map to 3D points and apply transformation
        # Formula (5), (9): convert Zdir to Pdir, then apply A to get Pglo
        B, C, H, W = depth_map.shape
        cu = camera_intrinsics['cx']
        cv = camera_intrinsics['cy']
        f = camera_intrinsics['focal_length']
        
        # Create pixel coordinate grids
        u = torch.arange(W, dtype=torch.float32, device=depth_map.device)
        v = torch.arange(H, dtype=torch.float32, device=depth_map.device)
        v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
        
        # Convert depth map to 3D points in camera coordinate system
        # Formula (5): x = (u - cu) * z / f, y = (v - cv) * z / f
        z = depth_map.squeeze(1)  # (B, H, W)
        x = (u_grid - cu) * z / (f + 1e-6)
        y = (v_grid - cv) * z / (f + 1e-6)
        
        # Stack into homogeneous coordinates: (x, y, z, 1)
        ones = torch.ones_like(z)
        Pdir = torch.stack([x, y, z, ones], dim=-1)  # (B, H, W, 4)
        
        # Apply transformation matrix to get Pglo
        # Formula (9): (x̄, ȳ, z̄) = (x, y, z) @ A
        Pglo = torch.matmul(Pdir, A.unsqueeze(-1)).squeeze(-1)  # (B, H, W, 4)
        
        # Extract reconstructed depth from global coordinates
        # Convert back to image coordinates
        x_bar = Pglo[..., 0]
        y_bar = Pglo[..., 1]
        z_bar = Pglo[..., 2]
        
        # Formula (5) inverse: reconstruct depth map
        global_depth = z_bar.unsqueeze(1)  # (B, 1, H, W)
        
        # Reliability score (equal to direct depth reliability)
        reliability = torch.ones_like(global_depth)
        
        return global_depth, reliability

    def _build_transformation_matrix(self, theta_R, theta_P, batch_size):
        """
        Build rotation matrices from roll and pitch angles
        Formula (7): AR = [[cosθR, -sinθR, 0],
                          [sinθR,  cosθR, 0],
                          [0,      0,     1]]
        
                    AP = [[1, 0,      0],
                          [0, cosθP, -sinθP],
                          [0, sinθP,  cosθP]]
        
        Formula (8): A = AR @ AP
        """
        cos_R = torch.cos(theta_R)
        sin_R = torch.sin(theta_R)
        cos_P = torch.cos(theta_P)
        sin_P = torch.sin(theta_P)
        
        device = theta_R.device
        dtype = theta_R.dtype
        
        # Build AR (roll rotation)
        AR = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        AR[:, 0, 0] = cos_R
        AR[:, 0, 1] = -sin_R
        AR[:, 1, 0] = sin_R
        AR[:, 1, 1] = cos_R
        AR[:, 2, 2] = 1.0
        
        # Build AP (pitch rotation)
        AP = torch.zeros(batch_size, 3, 3, dtype=dtype, device=device)
        AP[:, 0, 0] = 1.0
        AP[:, 1, 1] = cos_P
        AP[:, 1, 2] = -sin_P
        AP[:, 2, 1] = sin_P
        AP[:, 2, 2] = cos_P
        
        # Combine: A = AR @ AP
        A = torch.matmul(AR, AP)  # (B, 3, 3)
        
        return A
        
        # Create coordinate grids
        u_coords = torch.arange(W, dtype=torch.float32, device=Zdir.device)
        v_coords = torch.arange(H, dtype=torch.float32, device=Zdir.device)
        vv, uu = torch.meshgrid(v_coords, u_coords, indexing='ij')
        
        # Normalize coordinates
        x_norm = (uu - cu) * Zdir.squeeze(1) / self.focal_length  # (B, H, W)
        y_norm = (vv - cv) * Zdir.squeeze(1) / self.focal_length  # (B, H, W)
        z_norm = Zdir.squeeze(1)  # (B, H, W)
        
        # Stack to form 3D points
        Pdir = torch.stack([x_norm, y_norm, z_norm], dim=1)  # (B, 3, H, W)
        
        return Pdir
    
    def _transform_points(self, Pdir, A):
        """
        Transform 3D points using transformation matrix
        
        Formula (9): Pglo = Pdir @ A^T
        
        Args:
            Pdir: Input points (B, 3, H, W)
            A: Transformation matrix (3, 3)
            
        Returns:
            Pglo: Transformed points (B, 3, H, W)
        """
        batch_size, _, H, W = Pdir.shape
        
        # Reshape for matrix multiplication
        Pdir_flat = Pdir.permute(0, 2, 3, 1).reshape(-1, 3)  # (B*H*W, 3)
        
        # Apply transformation: (x, y, z) @ A^T
        Pglo_flat = torch.mm(Pdir_flat, A.t())  # (B*H*W, 3)
        
        # Reshape back
        Pglo = Pglo_flat.reshape(batch_size, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        
        return Pglo
    
    def _points_to_depth(self, Pglo, cu, cv):
        """
        Convert 3D points back to depth map
        
        Inverse of Formula (5):
            zuv = z
            u = x * f / z + cu
            v = y * f / z + cv
        
        Args:
            Pglo: 3D global points (B, 3, H, W)
            cu, cv: Principal point
            
        Returns:
            Zglo: Reconstructed depth map (B, 1, H, W)
        """
        # Extract coordinates
        x_glo = Pglo[:, 0, :, :]  # (B, H, W)
        y_glo = Pglo[:, 1, :, :]  # (B, H, W)
        z_glo = Pglo[:, 2, :, :]  # (B, H, W)
        
        # Depth is the z-coordinate
        Zglo = z_glo.unsqueeze(1)  # (B, 1, H, W)
        
        return Zglo
    
    def _get_device(self):
        """Get device of module parameters"""
        return next(self.parameters()).device
