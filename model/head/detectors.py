"""
Detectors Collection for VP-MonoMF

Includes:
- Depth Detector: Estimates depth map Zdir and reliability σdir
- Dimension Detector: Predicts 3D properties (length, width, height)
- Pose Detector: Estimates camera pose (horizon line, vanishing point)
- 2D Detector: Estimates 2D object properties (offset, center, orientation, height)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthDetector(nn.Module):
    """
    Depth Detector
    
    Estimates direct depth map Zdir and reliability confidence σdir
    from backbone features.
    """
    
    def __init__(self, in_channels, out_channels=2):
        """
        Args:
            in_channels: Number of input channels from backbone
            out_channels: Output channels (2 for depth + confidence)
        """
        super(DepthDetector, self).__init__()
        
        # Depth prediction head
        self.depth_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1)  # Depth map
        )
        
        # Confidence prediction head
        self.confidence_head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),  # Confidence
            nn.Sigmoid()  # Output in (0, 1)
        )
    
    def forward(self, features):
        """
        Args:
            features: Feature map from backbone (B, C, H, W)
            
        Returns:
            Zdir: Depth map (B, 1, H, W)
            sigma_dir: Reliability confidence (B, 1, H, W)
        """
        Zdir = self.depth_head(features)  # (B, 1, H, W)
        # Add small epsilon to avoid log(0)
        sigma_dir = torch.clamp(self.confidence_head(features), min=1e-6, max=1.0)  # (B, 1, H, W)
        
        return Zdir, sigma_dir


class DimensionDetector(nn.Module):
    """
    Dimension Detector
    
    Predicts 3D object dimensions (length, width, height) for each detected object.
    """
    
    def __init__(self, in_channels, max_objects=40):
        """
        Args:
            in_channels: Number of input channels from backbone
            max_objects: Maximum number of objects per image
        """
        super(DimensionDetector, self).__init__()
        self.max_objects = max_objects
        
        # Feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Dimension prediction (3 values: length, width, height)
        self.dim_predictor = nn.Conv2d(in_channels // 4, 3, kernel_size=1)
    
    def forward(self, features, targets=None):
        """
        Args:
            features: Feature map from backbone (B, C, H, W)
            targets: Ground truth targets with object information
            
        Returns:
            dimensions: 3D dimensions {dL, dW, dH} (B, N, 3)
                where dL=length, dW=width, dH=height
        """
        feat = self.feature_extract(features)  # (B, C', H, W)
        dim_map = self.dim_predictor(feat)  # (B, 3, H, W)
        
        # Extract positive dimensions using ReLU
        # Typically length, width, height should be positive
        dimensions = F.relu(dim_map.permute(0, 2, 3, 1))  # (B, H, W, 3)
        
        # For training, extract dimensions at object locations from targets
        # For inference, return spatially dense predictions
        if targets is not None and isinstance(targets, dict) and 'bboxes_2d' in targets:
            # Extract at object centers - placeholder
            batch_size = features.shape[0]
            dimensions_out = torch.zeros(batch_size, self.max_objects, 3, device=features.device)
            # In practice, would sample from dim_map at object center locations
            return dimensions_out
        else:
            # Return spatial map, can be aggregated later
            return dimensions


class PoseDetector(nn.Module):
    """
    Pose Detector
    
    Estimates camera pose parameters:
    - Horizon line (y = ax + b)
    - Vanishing point (xvp, yvp)
    """
    
    def __init__(self, in_channels):
        """
        Args:
            in_channels: Number of input channels from backbone
        """
        super(PoseDetector, self).__init__()
        
        # Feature extraction for pose estimation
        self.pose_feature = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Horizon line parameters: (a, b)
        self.horizon_predictor = nn.Sequential(
            nn.Linear(in_channels // 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Output: (a, b)
        )
        
        # Vanishing point: (xvp, yvp)
        self.vanishing_point_predictor = nn.Sequential(
            nn.Linear(in_channels // 4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)  # Output: (xvp, yvp)
        )
    
    def forward(self, features, img_width=None, img_height=None):
        """
        Args:
            features: Feature map from backbone (B, C, H, W)
            img_width: Image width for normalizing vanishing point
            img_height: Image height for normalizing vanishing point
            
        Returns:
            pose_info: Dict with 'horizon_params' and 'vanishing_point'
        """
        B, C, H, W = features.shape
        if img_width is None:
            img_width = W * 2
        if img_height is None:
            img_height = H * 2
        
        # Extract features
        pose_feat = self.pose_feature(features)  # (B, C', 1, 1)
        pose_feat = pose_feat.view(B, -1)  # (B, C')
        
        # Predict horizon line parameters
        horizon_params = self.horizon_predictor(pose_feat)  # (B, 2)
        
        # Predict vanishing point
        vp = self.vanishing_point_predictor(pose_feat)  # (B, 2)
        # Normalize vanishing point to image coordinates
        xvp = vp[:, 0] * img_width / 2.0 + img_width / 2.0
        yvp = vp[:, 1] * img_height / 2.0 + img_height / 2.0
        
        pose_info = {
            'horizon_params': horizon_params,  # (B, 2)
            'vanishing_point': torch.stack([xvp, yvp], dim=1)  # (B, 2)
        }
        
        return pose_info


class TwoDDetector(nn.Module):
    """
    2D Object Detector
    
    Estimates 2D object properties:
    - Object center offset
    - Center point
    - Orientation
    - 2D height and width
    """
    
    def __init__(self, in_channels, num_classes=3):
        """
        Args:
            in_channels: Number of input channels from backbone
            num_classes: Number of object classes (e.g., Car, Pedestrian, Cyclist)
        """
        super(TwoDDetector, self).__init__()
        self.num_classes = num_classes
        
        # Shared feature extraction
        self.feature_extract = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Center offset prediction
        self.offset_predictor = nn.Conv2d(in_channels // 4, 2, kernel_size=1)  # (dx, dy)
        
        # Center point heatmap
        self.center_predictor = nn.Conv2d(in_channels // 4, 1, kernel_size=1)
        
        # Orientation prediction (multi-bin)
        self.orientation_predictor = nn.Conv2d(in_channels // 4, 16, kernel_size=1)  # 8 bins × 2 (sin, cos)
        
        # 2D height and width prediction
        self.height_width_predictor = nn.Conv2d(in_channels // 4, 2, kernel_size=1)  # (h, w)
        
        # Classification scores
        self.class_predictor = nn.Conv2d(in_channels // 4, num_classes, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: Feature map from backbone (B, C, H, W)
            
        Returns:
            detection_dict: Dict with 2D detection results
                - 'offset': (B, 2, H, W)
                - 'center': (B, 1, H, W)
                - 'orientation': (B, 16, H, W)
                - 'height_width': (B, 2, H, W)
                - 'class_scores': (B, num_classes, H, W)
        """
        feat = self.feature_extract(features)  # (B, C', H, W)
        
        offset = self.offset_predictor(feat)
        center = self.center_predictor(feat)
        orientation = self.orientation_predictor(feat)
        height_width = self.height_width_predictor(feat)
        class_scores = self.class_predictor(feat)
        
        detection_dict = {
            'offset': offset,
            'center': center,
            'orientation': orientation,
            'height_width': height_width,
            'class_scores': class_scores
        }
        
        return detection_dict


def build_depth_detector(in_channels):
    """Build Depth Detector"""
    return DepthDetector(in_channels)


def build_dimension_detector(in_channels, max_objects=40):
    """Build Dimension Detector"""
    return DimensionDetector(in_channels, max_objects)


def build_pose_detector(in_channels):
    """Build Pose Detector"""
    return PoseDetector(in_channels)


def build_2d_detector(in_channels, num_classes=3):
    """Build 2D Detector"""
    return TwoDDetector(in_channels, num_classes)
