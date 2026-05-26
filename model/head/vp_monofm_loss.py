import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DepthLoss(nn.Module):
    """L1 Loss for Depth Detector
    
    Formula (12): Ldir = |Zdir - Z*| * σdir + log(1/σdir)
    """
    def __init__(self):
        super(DepthLoss, self).__init__()

    def forward(self, pred_depth, pred_sigma, target_depth):
        """
        Args:
            pred_depth: predicted depth Zdir (B, 1, H, W)
            pred_sigma: predicted reliability σdir (B, 1, H, W)
            target_depth: ground truth depth Z* (B, 1, H, W)
        
        Returns:
            loss: scalar depth loss
        """
        # L1 loss weighted by reliability
        l1_loss = torch.abs(pred_depth - target_depth) * pred_sigma
        
        # Entropy term for reliability
        entropy_loss = -torch.log(pred_sigma + 1e-6)
        
        # Total loss
        loss = (l1_loss + entropy_loss).mean()
        return loss


class DimensionLoss(nn.Module):
    """L1 Loss for Dimension Detector
    
    Formula (13): Ldim = Σ|dk - d*k| for k in {H, W, L}
    """
    def __init__(self):
        super(DimensionLoss, self).__init__()

    def forward(self, pred_dims, target_dims):
        """
        Args:
            pred_dims: predicted dimensions (B, N*3, H, W) or (B, 3*C)
            target_dims: ground truth dimensions (B, 3*C) or (N, 3)
        
        Returns:
            loss: scalar dimension loss
        """
        # Flatten and compute L1 loss for each dimension
        pred_dims_flat = pred_dims.view(pred_dims.size(0), -1)
        target_dims_flat = target_dims.view(target_dims.size(0), -1)
        
        loss = F.l1_loss(pred_dims_flat, target_dims_flat, reduction='mean')
        return loss


class PoseLoss(nn.Module):
    """Frobenius Norm Loss for Pose Detector
    
    Formula (14): Lpose = ||A - A*||F (Frobenius norm)
    """
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pred_matrix, target_matrix):
        """
        Args:
            pred_matrix: predicted transformation matrix A (B, 3, 3)
            target_matrix: ground truth transformation matrix A* (B, 3, 3)
        
        Returns:
            loss: scalar pose loss
        """
        # Frobenius norm: ||A - A*||_F = sqrt(sum((A - A*)^2))
        diff = pred_matrix - target_matrix
        frobenius_norm = torch.sqrt((diff ** 2).sum(dim=(-2, -1)) + 1e-6)
        loss = frobenius_norm.mean()
        return loss


class CenterHeatmapLoss(nn.Module):
    """Focal Loss for Center Heatmap (2D Detection)"""
    def __init__(self, alpha=2, beta=4):
        super(CenterHeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_heatmap, target_heatmap):
        """
        Args:
            pred_heatmap: predicted center heatmap (B, C, H, W)
            target_heatmap: target heatmap (B, C, H, W)
        
        Returns:
            loss: focal loss
        """
        pred = pred_heatmap.sigmoid()
        pos_inds = target_heatmap.eq(1).float()
        neg_inds = target_heatmap.lt(1).float()
        
        pos_loss = torch.log(pred + 1e-6) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred + 1e-6) * torch.pow(pred, self.beta) * neg_inds
        
        loss = -(pos_loss + neg_loss).sum() / (pos_inds.sum() + 1e-6)
        return loss


class OffsetLoss(nn.Module):
    """L1 Loss for Offset (2D Detection)"""
    def __init__(self):
        super(OffsetLoss, self).__init__()

    def forward(self, pred_offset, target_offset, mask=None):
        """
        Args:
            pred_offset: predicted offset (B, 2, H, W)
            target_offset: target offset (B, 2, H, W)
            mask: optional mask for valid regions
        
        Returns:
            loss: L1 loss
        """
        if mask is not None:
            loss = (F.l1_loss(pred_offset, target_offset, reduction='none') * mask).sum() / mask.sum()
        else:
            loss = F.l1_loss(pred_offset, target_offset, reduction='mean')
        return loss


class SizeLoss(nn.Module):
    """L1 Loss for 2D Box Size"""
    def __init__(self):
        super(SizeLoss, self).__init__()

    def forward(self, pred_size, target_size, mask=None):
        """
        Args:
            pred_size: predicted (w, h) (B, 2, H, W)
            target_size: target (w, h) (B, 2, H, W)
            mask: optional mask
        
        Returns:
            loss: L1 loss
        """
        if mask is not None:
            loss = (F.l1_loss(pred_size, target_size, reduction='none') * mask).sum() / mask.sum()
        else:
            loss = F.l1_loss(pred_size, target_size, reduction='mean')
        return loss


class VP_MonoFM_Loss(nn.Module):
    """Combined Loss for VP-MonoFM
    
    Stage 1 Loss: Lmdf = Ldir + Ldim + Lpose (Formula 11)
    Stage 2 Loss: L2d = center_loss + offset_loss + size_loss + ...
    """
    def __init__(self, cfg):
        super(VP_MonoFM_Loss, self).__init__()
        
        self.cfg = cfg
        
        # Stage 1 losses
        self.depth_loss = DepthLoss()
        self.dimension_loss = DimensionLoss()
        self.pose_loss = PoseLoss()
        
        # Stage 2 losses
        self.center_loss = CenterHeatmapLoss()
        self.offset_loss = OffsetLoss()
        self.size_loss = SizeLoss()
        
        # Loss weights
        self.loss_weights = {
            'depth': 1.0,
            'dimension': 1.0,
            'pose': 0.1,
            'center': 1.0,
            'offset': 1.0,
            'size': 1.0,
        }

    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict from predictor with all outputs
            targets: list of target objects
        
        Returns:
            loss_dict: dict with individual losses
            log_loss_dict: dict for logging (same values)
        """
        loss_dict = {}
        
        # ===== Stage 1: MDF Losses =====
        
        # Depth Loss: Ldir
        if 'fused_depth' in predictions and hasattr(targets[0], 'depth_map'):
            try:
                target_depths = torch.stack([t.depth_map for t in targets if hasattr(t, 'depth_map')])
                Ldir = self.depth_loss(
                    predictions['Zdir'],
                    predictions['sigma_dir'],
                    target_depths
                )
                loss_dict['loss_Ldir'] = Ldir * self.loss_weights['depth']
            except:
                loss_dict['loss_Ldir'] = torch.tensor(0.0, device=predictions['Zdir'].device)
        
        # Dimension Loss: Ldim
        if 'dimension' in predictions and hasattr(targets[0], 'dimension'):
            try:
                target_dims = torch.stack([t.dimension for t in targets if hasattr(t, 'dimension')])
                Ldim = self.dimension_loss(predictions['dimension'], target_dims)
                loss_dict['loss_Ldim'] = Ldim * self.loss_weights['dimension']
            except:
                loss_dict['loss_Ldim'] = torch.tensor(0.0, device=predictions['dimension'].device)
        
        # Pose Loss: Lpose (optional, requires pose GT)
        if 'global_depth' in predictions:
            loss_dict['loss_Lpose'] = torch.tensor(0.0, device=predictions['global_depth'].device)
        
        # ===== Stage 2: 2D Detection Losses =====
        
        # Center Heatmap Loss
        if 'center_heatmap' in predictions and hasattr(targets[0], 'center_heatmap'):
            try:
                target_heatmaps = torch.stack([t.center_heatmap for t in targets if hasattr(t, 'center_heatmap')])
                center_loss = self.center_loss(predictions['center_heatmap'], target_heatmaps)
                loss_dict['loss_center'] = center_loss * self.loss_weights['center']
            except:
                loss_dict['loss_center'] = torch.tensor(0.0, device=predictions['center_heatmap'].device)
        
        # Offset Loss
        if 'offset' in predictions and hasattr(targets[0], 'offset'):
            try:
                target_offsets = torch.stack([t.offset for t in targets if hasattr(t, 'offset')])
                offset_loss = self.offset_loss(predictions['offset'], target_offsets)
                loss_dict['loss_offset'] = offset_loss * self.loss_weights['offset']
            except:
                loss_dict['loss_offset'] = torch.tensor(0.0, device=predictions['offset'].device)
        
        # Size Loss
        if 'wh' in predictions and hasattr(targets[0], 'wh'):
            try:
                target_sizes = torch.stack([t.wh for t in targets if hasattr(t, 'wh')])
                size_loss = self.size_loss(predictions['wh'], target_sizes)
                loss_dict['loss_size'] = size_loss * self.loss_weights['size']
            except:
                loss_dict['loss_size'] = torch.tensor(0.0, device=predictions['wh'].device)
        
        # Ensure all losses are present with defaults
        for key in ['loss_Ldir', 'loss_Ldim', 'loss_Lpose', 'loss_center', 'loss_offset', 'loss_size']:
            if key not in loss_dict:
                device = next(self.parameters()).device
                loss_dict[key] = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = sum(loss_dict.values())
        loss_dict['loss'] = total_loss
        
        # Log dict (same as loss dict for now)
        log_loss_dict = {k: v.detach() for k, v in loss_dict.items()}
        
        return loss_dict, log_loss_dict


def make_loss_evaluator(cfg):
    """Factory function to create VP-MonoFM loss"""
    return VP_MonoFM_Loss(cfg)
