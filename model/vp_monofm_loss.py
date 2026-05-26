"""
VP-MonoMF Loss Functions

Implements:
- Stage 1 Loss: LMDF = Ldir + Ldim + Lpose (Formula 11-14)
- Stage 2 Loss: L2D for 2D detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    """
    Depth Detection Loss
    
    Formula (12): Ldir = |Zdir - Z*| × σdir + log(1/σdir)
    
    Combines L1 loss weighted by confidence and uncertainty regularization
    """
    
    def __init__(self):
        super(DepthLoss, self).__init__()
    
    def forward(self, Zdir, sigma_dir, Z_gt):
        """
        Args:
            Zdir: Predicted depth map (B, 1, H, W)
            sigma_dir: Reliability confidence (B, 1, H, W), in (0, 1]
            Z_gt: Ground truth depth map (B, 1, H, W)
            
        Returns:
            loss: Scalar loss value
        """
        # L1 loss weighted by confidence
        l1_loss = torch.abs(Zdir - Z_gt) * sigma_dir  # (B, 1, H, W)
        
        # Uncertainty regularization: log(1/σdir) = -log(σdir)
        uncertainty_loss = -torch.log(sigma_dir)  # (B, 1, H, W)
        
        # Combined loss
        loss = (l1_loss + uncertainty_loss).mean()
        
        return loss


class DimensionLoss(nn.Module):
    """
    Dimension Detection Loss
    
    Formula (13): Ldim = Σ|dk - d*k| for k ∈ {H, W, L}
    
    L1 loss on predicted vs ground truth 3D dimensions
    """
    
    def __init__(self):
        super(DimensionLoss, self).__init__()
    
    def forward(self, dimensions, dimensions_gt):
        """
        Args:
            dimensions: Predicted dimensions (B, N, 3) or (B, H, W, 3)
                       3 channels: [height, width, length]
            dimensions_gt: Ground truth dimensions (B, N, 3)
            
        Returns:
            loss: Scalar loss value
        """
        if dimensions is None or dimensions_gt is None:
            return torch.tensor(0.0, device=dimensions.device if dimensions is not None else dimensions_gt.device)
        
        # L1 loss on all three dimensions
        loss = F.l1_loss(dimensions, dimensions_gt)
        
        return loss


class PoseLoss(nn.Module):
    """
    Pose Detection Loss
    
    Formula (14): Lpose = ||A - A*||_F
    
    Frobenius norm of difference between predicted and GT transformation matrices
    """
    
    def __init__(self):
        super(PoseLoss, self).__init__()
    
    def forward(self, A_pred, A_gt):
        """
        Args:
            A_pred: Predicted transformation matrix (B, 3, 3)
            A_gt: Ground truth transformation matrix (B, 3, 3)
            
        Returns:
            loss: Scalar loss value
        """
        if A_pred is None or A_gt is None:
            return torch.tensor(0.0, device=A_pred.device if A_pred is not None else A_gt.device)
        
        # Frobenius norm: sqrt(sum of squared differences)
        diff = A_pred - A_gt  # (B, 3, 3)
        frobenius_norm = torch.norm(diff, p='fro', dim=(1, 2))  # (B,)
        loss = frobenius_norm.mean()
        
        return loss


class VPMonoMFLoss(nn.Module):
    """
    VP-MonoMF Loss Function
    
    Stage 1: LMDF = Ldir + Ldim + Lpose
    Stage 2: L2D (2D detection losses)
    
    Total: L_total = LMDF + λ × L2D
    """
    
    def __init__(self, cfg):
        super(VPMonoMFLoss, self).__init__()
        self.cfg = cfg
        
        # Stage 1 losses
        self.depth_loss = DepthLoss()
        self.dimension_loss = DimensionLoss()
        self.pose_loss = PoseLoss()
        
        # Loss weights
        self.weight_depth = 1.0
        self.weight_dimension = 1.0
        self.weight_pose = 1.0
        self.weight_2d = 1.0
        
        # Extract weights from config if available
        if hasattr(cfg, 'MODEL') and hasattr(cfg.MODEL, 'LOSS_WEIGHTS'):
            weights = cfg.MODEL.LOSS_WEIGHTS
            self.weight_depth = weights.get('depth', 1.0)
            self.weight_dimension = weights.get('dimension', 1.0)
            self.weight_pose = weights.get('pose', 1.0)
            self.weight_2d = weights.get('2d', 1.0)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: Dict with model predictions from forward pass
            targets: Dict with ground truth annotations
            
        Returns:
            loss_dict: Dict with individual losses
            log_loss_dict: Dict with losses for logging
        """
        loss_dict = {}
        log_loss_dict = {}
        
        # ============ Stage 1: Depth-based losses ============
        
        # Depth loss (Formula 12)
        if 'Zdir' in predictions and targets.get('Z_gt') is not None:
            L_dir = self.depth_loss(
                predictions['Zdir'],
                predictions.get('sigma_dir'),
                targets['Z_gt']
            )
            loss_dict['loss_depth'] = L_dir * self.weight_depth
            log_loss_dict['loss_depth'] = L_dir.item()
        else:
            loss_dict['loss_depth'] = torch.tensor(0.0)
        
        # Dimension loss (Formula 13)
        if predictions.get('dimensions') is not None and targets.get('dimensions_gt') is not None:
            L_dim = self.dimension_loss(
                predictions['dimensions'],
                targets['dimensions_gt']
            )
            loss_dict['loss_dimension'] = L_dim * self.weight_dimension
            log_loss_dict['loss_dimension'] = L_dim.item()
        else:
            loss_dict['loss_dimension'] = torch.tensor(0.0)
        
        # Pose loss (Formula 14)
        if predictions.get('pose_dict') is not None and targets.get('pose_dict_gt') is not None:
            A_pred = predictions['pose_dict'].get('transformation_matrix')
            A_gt = targets['pose_dict_gt'].get('transformation_matrix')
            
            if A_pred is not None and A_gt is not None:
                L_pose = self.pose_loss(A_pred, A_gt)
                loss_dict['loss_pose'] = L_pose * self.weight_pose
                log_loss_dict['loss_pose'] = L_pose.item()
            else:
                loss_dict['loss_pose'] = torch.tensor(0.0)
        else:
            loss_dict['loss_pose'] = torch.tensor(0.0)
        
        # ============ Stage 2: 2D detection losses ============
        
        # Center/heatmap loss
        if 'center' in predictions and targets.get('center_heatmap') is not None:
            center_loss = F.binary_cross_entropy_with_logits(
                predictions['center'],
                targets['center_heatmap']
            )
            loss_dict['loss_center'] = center_loss
            log_loss_dict['loss_center'] = center_loss.item()
        else:
            loss_dict['loss_center'] = torch.tensor(0.0)
        
        # Offset loss
        if 'offset' in predictions and targets.get('offset_gt') is not None:
            offset_loss = F.l1_loss(predictions['offset'], targets['offset_gt'])
            loss_dict['loss_offset'] = offset_loss
            log_loss_dict['loss_offset'] = offset_loss.item()
        else:
            loss_dict['loss_offset'] = torch.tensor(0.0)
        
        # Orientation loss
        if 'orientation' in predictions and targets.get('orientation_gt') is not None:
            orientation_loss = F.smooth_l1_loss(predictions['orientation'], targets['orientation_gt'])
            loss_dict['loss_orientation'] = orientation_loss
            log_loss_dict['loss_orientation'] = orientation_loss.item()
        else:
            loss_dict['loss_orientation'] = torch.tensor(0.0)
        
        # Height/width loss
        if 'height_width' in predictions and targets.get('height_width_gt') is not None:
            hw_loss = F.smooth_l1_loss(predictions['height_width'], targets['height_width_gt'])
            loss_dict['loss_height_width'] = hw_loss
            log_loss_dict['loss_height_width'] = hw_loss.item()
        else:
            loss_dict['loss_height_width'] = torch.tensor(0.0)
        
        # Classification loss
        if 'class_scores' in predictions and targets.get('class_labels') is not None:
            class_loss = F.cross_entropy(predictions['class_scores'], targets['class_labels'])
            loss_dict['loss_classification'] = class_loss
            log_loss_dict['loss_classification'] = class_loss.item()
        else:
            loss_dict['loss_classification'] = torch.tensor(0.0)
        
        return loss_dict, log_loss_dict


def build_vp_monofm_loss(cfg):
    """Build VP-MonoMF loss function"""
    return VPMonoMFLoss(cfg)
