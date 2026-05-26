import torch
import torch.nn as nn
from .mdf_module import MDF
from .threed_reconstruction import ThreeD_Reconstruction
from .vpf_module import VPF


class TwoD_Detector(nn.Module):
    """2D Object Detector
    
    Estimates 2D properties: offset, center, orientation, height
    """
    def __init__(self, in_channels, num_classes=3):
        super(TwoD_Detector, self).__init__()
        self.num_classes = num_classes
        
        # Shared backbone for 2D detection
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Different heads for different 2D properties
        self.offset_head = nn.Conv2d(64, 2, 1)  # (x, y) offset
        self.center_head = nn.Conv2d(64, num_classes, 1)  # object center heatmap
        self.wh_head = nn.Conv2d(64, 2, 1)  # (w, h) of 2D bbox
        self.height_head = nn.Conv2d(64, 1, 1)  # 2D height of object
        self.orientation_head = nn.Conv2d(64, 8, 1)  # orientation (8 for multi-bin)

    def forward(self, features):
        """
        Returns:
            offset: (B, 2, H, W)
            center: (B, C, H, W) - class heatmap
            wh: (B, 2, H, W)
            height: (B, 1, H, W)
            orientation: (B, 8, H, W)
        """
        x = self.conv(features)
        
        offset = self.offset_head(x)
        center = self.center_head(x)
        wh = self.wh_head(x)
        height = self.height_head(x)
        orientation = self.orientation_head(x)
        
        return {
            'offset': offset,
            'center': center,
            'wh': wh,
            'height': height,
            'orientation': orientation
        }


class VP_MonoFM_Predictor(nn.Module):
    """Visual Prompt-guided Monocular 3D Object Detection with Multiscale Fusion
    
    Two-stage architecture:
    Stage 1: Extract depth information using MDF module
    Stage 2: Enhance features using VPF and estimate 2D properties
    """
    def __init__(self, cfg, in_channels):
        super(VP_MonoFM_Predictor, self).__init__()
        
        self.cfg = cfg
        self.in_channels = in_channels
        
        # Stage 1: Multiscale Depth Fusion
        self.mdf = MDF(in_channels, num_objects=cfg.DATASETS.MAX_OBJECTS)
        
        # 3D Reconstruction module
        self.threed_reconstruction = ThreeD_Reconstruction(in_channels)
        
        # Stage 2: Visual Prompt Feature enhancement
        self.vpf = VPF(in_channels)
        
        # 2D Detection head
        self.detector_2d = TwoD_Detector(in_channels, num_classes=len(cfg.DATASETS.DETECT_CLASSES))
        
        # Camera intrinsics (KITTI default)
        self.register_buffer(
            'camera_intrinsics',
            torch.tensor([721.5377, 609.5593, 172.8540, 1280.0, 384.0])
        )  # [f, cx, cy, W, H]

    def forward(self, features, targets=None):
        """
        Args:
            features: backbone features (B, C, H, W)
            targets: training targets with GT information
        
        Returns:
            predictions: dict with all outputs from both stages
        """
        B, C, H, W = features.shape
        
        # ===== Stage 1: Multiscale Depth Fusion =====
        
        # Get 3DR global depth
        fused_depth, dims = None, None
        try:
            global_depth, global_reliability = self.threed_reconstruction(
                torch.zeros(B, 1, H, W, device=features.device),
                features
            )
        except:
            global_depth = None
            global_reliability = None
        
        # Prepare camera intrinsics
        camera_intrinsics = {
            'focal_length': float(self.camera_intrinsics[0].item()),
            'cx': float(self.camera_intrinsics[1].item()),
            'cy': float(self.camera_intrinsics[2].item()),
        }
        
        # MDF: fuse global and local depths
        fused_depth, dims, Zdir, sigma_dir = self.mdf(
            features,
            height_2d=None,  # Would come from GT or 2D detector
            focal_length=camera_intrinsics['focal_length'],
            global_depth=global_depth,
            global_reliability=global_reliability
        )
        
        # ===== Stage 2: Visual Prompt Feature Enhancement =====
        
        # Prepare 2D bounding boxes and sizes for VPF (from targets if training)
        bboxes_2d = None
        object_sizes = None
        if self.training and targets is not None:
            # Extract 2D bboxes from targets
            if hasattr(targets[0], 'bbox_2d'):
                # Combine all bboxes from batch
                bboxes_2d_list = []
                sizes_list = []
                for target in targets:
                    if hasattr(target, 'bbox_2d') and len(target.bbox_2d) > 0:
                        bboxes_2d_list.append(target.bbox_2d)
                        # Calculate sizes (area)
                        bbox = target.bbox_2d
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        sizes_list.append(area)
                
                if bboxes_2d_list:
                    bboxes_2d = torch.cat(bboxes_2d_list, dim=0)
                    object_sizes = torch.tensor(sizes_list, device=features.device)
        
        # VPF: enhance features
        enhanced_features, attention_map = self.vpf(
            features,
            bboxes_2d=bboxes_2d,
            object_sizes=object_sizes,
            training=self.training
        )
        
        # ===== 2D Detection =====
        detections_2d = self.detector_2d(enhanced_features)
        
        # Compile all predictions
        predictions = {
            # Stage 1 outputs
            'fused_depth': fused_depth,
            'Zdir': Zdir,
            'sigma_dir': sigma_dir,
            'dimension': dims,
            'global_depth': global_depth,
            'global_reliability': global_reliability,
            
            # Stage 2 outputs
            'attention_map': attention_map,
            
            # 2D detection outputs
            'offset': detections_2d['offset'],
            'center_heatmap': detections_2d['center'],
            'wh': detections_2d['wh'],
            'height_2d': detections_2d['height'],
            'orientation': detections_2d['orientation'],
        }
        
        return predictions


def make_predictor(cfg, in_channels):
    """Factory function to create VP-MonoFM predictor"""
    return VP_MonoFM_Predictor(cfg, in_channels)
