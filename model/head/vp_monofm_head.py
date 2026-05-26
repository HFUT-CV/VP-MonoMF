"""
VP-MonoMF Detection Head

Main architecture combining all modules:
- Stage 1: MDF (with Depth, Dimension, Pose detectors)
- Stage 2: 2D detection with VPF enhancement
"""

import torch
import torch.nn as nn

from .detectors import (
    build_depth_detector,
    build_dimension_detector,
    build_pose_detector,
    build_2d_detector
)
from .mdf_module import MDFModule
from .threed_reconstruction import ThreeDReconstruction
from .vpf_module import VPFModule


class VPMonoMFHead(nn.Module):
    """
    VP-MonoMF Detection Head
    
    Two-stage architecture:
    Stage 1: Extract depth information via MDF module
    - Depth Detector: Zdir and σdir
    - Dimension Detector: 3D dimensions
    - Pose Detector: Camera pose
    - 3DR Module: Global depth correction
    
    Stage 2: Enhance features and detect 2D properties
    - VPF Module: Visual prompt feature enhancement
    - 2D Detector: 2D object properties
    """
    
    def __init__(self, cfg, in_channels):
        super(VPMonoMFHead, self).__init__()
        self.cfg = cfg
        self.in_channels = in_channels
        
        # =============== Stage 1: Depth-based modules ===============
        # Detectors
        self.depth_detector = build_depth_detector(in_channels)
        self.dimension_detector = build_dimension_detector(in_channels)
        self.pose_detector = build_pose_detector(in_channels)
        
        # 3D Reconstruction module
        self.threed_reconstruction = ThreeDReconstruction(cfg)
        
        # Multiscale Depth Fusion module
        self.mdf_module = MDFModule(cfg)
        
        # =============== Stage 2: 2D detection modules ===============
        # Visual Prompt Feature module
        self.vpf_module = VPFModule(cfg)
        
        # 2D Detector
        num_classes = len(cfg.DATASETS.DETECT_CLASSES) if hasattr(cfg.DATASETS, 'DETECT_CLASSES') else 3
        self.detector_2d = build_2d_detector(in_channels, num_classes)
        
    def forward(self, features, targets=None):
        """
        Forward pass for VP-MonoMF
        
        Args:
            features: Feature maps from backbone (B, C, H, W)
            targets: Ground truth targets (training only)
            
        Returns:
            If training:
                predictions: Dict with all prediction outputs
                targets: Modified targets if needed
            If inference:
                results: Detection results
        """
        
        if self.training:
            return self._forward_train(features, targets)
        else:
            return self._forward_test(features, targets)
    
    def _forward_train(self, features, targets):
        """
        Training forward pass
        
        Returns:
            predictions: Dict with stage 1 and stage 2 outputs
            targets: Input targets
        """
        predictions = {}
        
        # ============ Stage 1: Depth Extraction ============
        # Depth Detector
        Zdir, sigma_dir = self.depth_detector(features)
        predictions['Zdir'] = Zdir
        predictions['sigma_dir'] = sigma_dir
        
        # Pose Detector
        pose_info = self.pose_detector(features)
        predictions['pose_info'] = pose_info
        
        # 3D Reconstruction (global depth)
        Zglo, sigma_glo, pose_dict = self.threed_reconstruction(
            features, Zdir, sigma_dir, 
            pose_info=pose_info,
            targets=targets
        )
        predictions['Zglo'] = Zglo
        predictions['sigma_glo'] = sigma_glo
        predictions['pose_dict'] = pose_dict
        
        # Dimension Detector
        dimensions = self.dimension_detector(features, targets)
        predictions['dimensions'] = dimensions
        
        # Prepare keypoint properties for MDF
        keypoint_properties = {
            'keypoints_2d': targets.get('keypoints', None) if isinstance(targets, dict) else None,
            'heights_2d': targets.get('heights_2d', None) if isinstance(targets, dict) else None,
            'sigma_heights': targets.get('sigma_heights', None) if isinstance(targets, dict) else None,
            'dimensions': dimensions
        }
        
        # MDF: Fuse depths
        Zcom, dim_out = self.mdf_module(
            Zglo, sigma_glo,
            {'Zdir': Zdir, 'sigma_dir': sigma_dir},
            keypoint_properties,
            targets=targets
        )
        predictions['Zcom'] = Zcom
        predictions['dim_mdf'] = dim_out
        
        # ============ Stage 2: 2D Detection with VPF ============
        # Visual Prompt Feature enhancement
        features_enhanced, visual_prompt = self.vpf_module(
            features, targets=targets, test=False
        )
        predictions['visual_prompt'] = visual_prompt
        
        # 2D Detector on enhanced features
        detections_2d = self.detector_2d(features_enhanced)
        predictions.update(detections_2d)
        
        return predictions, targets
    
    def _forward_test(self, features, targets):
        """
        Inference forward pass
        
        Returns:
            results: List of detection results
            visualize_preds: Visualization data
        """
        results = []
        visualize_preds = {}
        
        # ============ Stage 1: Depth Extraction ============
        Zdir, sigma_dir = self.depth_detector(features)
        
        pose_info = self.pose_detector(features)
        
        Zglo, sigma_glo, pose_dict = self.threed_reconstruction(
            features, Zdir, sigma_dir,
            pose_info=pose_info,
            targets=None
        )
        
        dimensions = self.dimension_detector(features, targets=None)
        
        keypoint_properties = {
            'dimensions': dimensions
        }
        
        Zcom, _ = self.mdf_module(
            Zglo, sigma_glo,
            {'Zdir': Zdir, 'sigma_dir': sigma_dir},
            keypoint_properties,
            targets=None
        )
        
        # ============ Stage 2: 2D Detection without VPF ============
        # During inference, no visual prompt
        features_enhanced, _ = self.vpf_module(
            features, targets=None, test=True
        )
        
        # 2D Detection
        detections_2d = self.detector_2d(features_enhanced)
        
        # Post-processing and NMS would happen here
        # Convert raw predictions to final detections
        
        visualize_preds = {
            'Zdir': Zdir,
            'Zcom': Zcom,
            'pose_dict': pose_dict
        }
        
        return results, {}, visualize_preds


def build_vp_monofm_head(cfg, in_channels):
    """Build VP-MonoMF detection head"""
    return VPMonoMFHead(cfg, in_channels)
