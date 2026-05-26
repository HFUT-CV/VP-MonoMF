import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VP_MonoFM_PostProcessor(nn.Module):
    """Post-processor for VP-MonoFM inference
    
    Converts network outputs to final 3D object detections
    """
    def __init__(self, cfg):
        super(VP_MonoFM_PostProcessor, self).__init__()
        self.cfg = cfg
        self.num_classes = len(cfg.DATASETS.DETECT_CLASSES)
        self.detect_classes = cfg.DATASETS.DETECT_CLASSES
        self.max_objects = cfg.DATASETS.MAX_OBJECTS
        
        # NMS parameters
        self.nms_threshold = 0.5
        self.score_threshold = 0.3

    def forward(self, predictions, targets=None, test=False, features=None):
        """
        Args:
            predictions: dict with network outputs
            targets: optional target objects (not used in inference)
            test: whether in test mode
            features: backbone features (optional)
        
        Returns:
            result: dict with 3D detections
            eval_utils: dict for evaluation
            visualize_preds: dict for visualization
        """
        batch_size = predictions['center_heatmap'].size(0)
        
        result = []
        eval_utils = []
        visualize_preds = []
        
        for b in range(batch_size):
            # Extract predictions for batch element
            center_heatmap = predictions['center_heatmap'][b]  # (C, H, W)
            offset = predictions['offset'][b]  # (2, H, W)
            wh = predictions['wh'][b]  # (2, H, W)
            height_2d = predictions['height_2d'][b]  # (1, H, W)
            fused_depth = predictions['fused_depth'][b]  # (1, H, W)
            dimension = predictions['dimension'][b]  # (N*3, H, W)
            orientation = predictions['orientation'][b]  # (8, H, W)
            
            # Peak detection on heatmap
            detections = self._decode_detections(
                center_heatmap, offset, wh, height_2d, 
                fused_depth, dimension, orientation
            )
            
            result.append(detections)
            
            # Prepare evaluation utilities
            eval_util = {
                'center_heatmap': center_heatmap.detach().cpu().numpy(),
                'fused_depth': fused_depth.detach().cpu().numpy(),
            }
            eval_utils.append(eval_util)
            
            # Prepare visualization predictions
            vis_pred = {
                'detections': detections,
                'heatmap': center_heatmap.detach().cpu().numpy(),
            }
            visualize_preds.append(vis_pred)
        
        return result, eval_utils, visualize_preds

    def _decode_detections(self, center_heatmap, offset, wh, height_2d,
                          fused_depth, dimension, orientation):
        """
        Decode network outputs to 3D detections
        
        Args:
            center_heatmap: (C, H, W)
            offset: (2, H, W)
            wh: (2, H, W)
            height_2d: (1, H, W)
            fused_depth: (1, H, W)
            dimension: (N*3, H, W)
            orientation: (8, H, W)
        
        Returns:
            detections: list of detected objects
        """
        detections = []
        
        H, W = center_heatmap.shape[-2:]
        
        # Find peaks in center heatmap
        for class_idx in range(center_heatmap.size(0)):
            heatmap = center_heatmap[class_idx]
            
            # Simple peak detection (can be improved with NMS)
            peaks = self._find_peaks(heatmap)
            
            for peak in peaks:
                y, x, score = peak
                
                if score < self.score_threshold:
                    continue
                
                # Get offset
                offset_xy = offset[:, y, x]  # (2,)
                center_x = x + offset_xy[0].item()
                center_y = y + offset_xy[1].item()
                
                # Get 2D box size
                bbox_wh = wh[:, y, x]  # (2,)
                bbox_w = bbox_wh[0].item()
                bbox_h = bbox_wh[1].item()
                
                # Get 3D height
                height_3d = height_2d[0, y, x].item()
                
                # Get depth
                depth = fused_depth[0, y, x].item()
                
                # Get dimensions
                dim_idx = class_idx % (dimension.size(0) // 3)
                dim_h = dimension[dim_idx * 3, y, x].item()
                dim_w = dimension[dim_idx * 3 + 1, y, x].item()
                dim_l = dimension[dim_idx * 3 + 2, y, x].item()
                
                # Get orientation
                orient_bin = orientation[:, y, x]
                orient_angle = self._decode_orientation(orient_bin)
                
                # Create detection
                detection = {
                    'class_id': class_idx,
                    'class_name': self.detect_classes[class_idx] if class_idx < len(self.detect_classes) else 'Unknown',
                    'score': float(score),
                    'bbox_2d': [center_x - bbox_w / 2, center_y - bbox_h / 2,
                               center_x + bbox_w / 2, center_y + bbox_h / 2],
                    'center_2d': [center_x, center_y],
                    'height_2d': float(height_3d),
                    'dimension_3d': [dim_h, dim_w, dim_l],
                    'depth': float(depth),
                    'orientation': float(orient_angle),
                }
                
                detections.append(detection)
        
        return detections

    def _find_peaks(self, heatmap, threshold=0.3, nms_kernel=3):
        """Find peaks in heatmap"""
        peaks = []
        
        # Threshold heatmap
        mask = heatmap > threshold
        coords = torch.where(mask)
        
        if len(coords[0]) == 0:
            return peaks
        
        for i in range(len(coords[0])):
            y, x = coords[0][i].item(), coords[1][i].item()
            score = heatmap[y, x].item()
            peaks.append((y, x, score))
        
        # Sort by score
        peaks = sorted(peaks, key=lambda p: p[2], reverse=True)
        
        # Keep only top detections
        peaks = peaks[:self.cfg.DATASETS.MAX_OBJECTS]
        
        return peaks

    def _decode_orientation(self, orient_bin):
        """Decode orientation from multi-bin representation"""
        # Simple approach: take argmax of 8-bin orientation
        # In practice, should use continuous representation
        max_idx = torch.argmax(orient_bin).item()
        angle = (max_idx * 2 * np.pi) / 8.0  # Convert bin to angle
        return angle


def make_post_processor(cfg):
    """Factory function to create post-processor"""
    return VP_MonoFM_PostProcessor(cfg)
