import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthDetector(nn.Module):
    """Detect depth map and reliability scores"""
    def __init__(self, in_channels, out_channels=1):
        super(DepthDetector, self).__init__()
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.depth_out = nn.Conv2d(32, out_channels, 1)
        self.reliability_out = nn.Sequential(
            nn.Conv2d(32, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        x = self.depth_conv(features)
        depth = self.depth_out(x)  # Zdir
        reliability = self.reliability_out(x)  # σdir
        return depth, reliability


class DimensionDetector(nn.Module):
    """Detect 3D object dimensions"""
    def __init__(self, in_channels, num_objects=40):
        super(DimensionDetector, self).__init__()
        self.num_objects = num_objects
        self.dim_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Output: length, width, height for each object
        self.dim_head = nn.Conv2d(32, num_objects * 3, 1)

    def forward(self, features):
        x = self.dim_conv(features)
        dims = self.dim_head(x)  # (B, N*3, H, W)
        return dims


class MDF(nn.Module):
    """Multiscale Depth Fusion Module
    
    Fuses global depth (from 3DR) and local depth (from 2D height + 3D dimension)
    """
    def __init__(self, in_channels, num_objects=40):
        super(MDF, self).__init__()
        self.num_objects = num_objects
        self.depth_detector = DepthDetector(in_channels)
        self.dimension_detector = DimensionDetector(in_channels, num_objects)

    def forward(self, features, height_2d=None, focal_length=None, 
                global_depth=None, global_reliability=None):
        """
        Args:
            features: backbone features (B, C, H, W)
            height_2d: 2D bounding box heights with reliability scores
            focal_length: camera focal length (scalar)
            global_depth: Zglo from 3DR module
            global_reliability: σglo from 3DR module
        
        Returns:
            fused_depth: Zcom (B, 1, H, W)
            dimension: object dimensions (B, N*3, H, W)
        """
        # Branch 1: Get global depth from Depth Detector + 3DR
        Zdir, sigma_dir = self.depth_detector(features)
        
        if global_depth is None:
            Zglo = Zdir
            sigma_glo = sigma_dir
        else:
            Zglo = global_depth
            sigma_glo = global_reliability

        # Branch 2: Get local depth from 2D height and 3D dimensions
        dim = self.dimension_detector(features)  # (B, N*3, H, W)
        
        # Calculate local depths using formula (1), (2), (3)
        if height_2d is not None and focal_length is not None:
            Zloc, sigma_loc = self._calculate_local_depth(
                dim, height_2d, focal_length, features.size()
            )
        else:
            # Fallback: use global depth as local depth
            Zloc = Zglo
            sigma_loc = sigma_glo

        # Fuse depths using formula (4): weighted average by reliability
        fused_depth = self._fuse_depths(Zglo, sigma_glo, Zloc, sigma_loc)

        return fused_depth, dim, Zdir, sigma_dir

    def _calculate_local_depth(self, dim, height_2d, focal_length, feat_size):
        """
        Calculate local depth from 2D height and 3D dimensions
        Formula (1): zi = f * dH / hi
        Formula (2), (3): average depths and reliability scores
        """
        B, C, H, W = feat_size
        
        # Extract 3D height from dimensions
        # dim shape: (B, N*3, H, W) - contains dL, dW, dH for each object
        # For simplicity, use the mean dimension across spatial dims
        dH = dim[:, 2::3, :, :].mean(dim=(-2, -1), keepdim=True)  # (B, N, 1, 1)
        
        # height_2d: dictionary with hi and σi for 4 edges and center
        # Simplified: calculate depth as f * dH / h_center
        if isinstance(height_2d, dict) and 'h_center' in height_2d:
            h_center = height_2d['h_center']
            sigma_center = height_2d.get('sigma_center', 1.0)
            
            # Avoid division by zero
            Zloc = (focal_length * dH) / (h_center + 1e-6)
            sigma_loc = torch.ones_like(Zloc) * sigma_center
        else:
            # Fallback
            Zloc = dH
            sigma_loc = torch.ones_like(dH)
        
        return Zloc, sigma_loc

    def _fuse_depths(self, Zglo, sigma_glo, Zloc, sigma_loc):
        """
        Fuse depths using weighted average
        Formula (4): Zcom = Σ(zk * σk) / Σ(σk)
        
        Combines: zd1, zd2, zc (from local), zx'y' (from global)
        """
        # Ensure all depths have same shape for broadcasting
        B, C, H, W = Zglo.shape
        
        # Normalize reliability scores to (0, 1]
        sigma_glo = torch.clamp(sigma_glo, min=1e-6, max=1.0)
        sigma_loc = torch.clamp(sigma_loc, min=1e-6, max=1.0)
        
        # Weighted fusion
        numerator = Zglo * sigma_glo + Zloc * sigma_loc
        denominator = sigma_glo + sigma_loc + 1e-6
        fused_depth = numerator / denominator
        
        return fused_depth
