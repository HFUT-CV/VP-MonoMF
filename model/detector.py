import torch
from torch import nn

from structures.image_list import to_image_list

from .backbone import build_backbone
from .head.vp_monofm_predictor import make_predictor
from .head.vp_monofm_loss import make_loss_evaluator
from .head.vp_monofm_inference import make_post_processor


class VP_MonoFM_Head(nn.Module):
    """VP-MonoFM Detection Head
    
    Combines MDF, 3DR, VPF and 2D detection for monocular 3D object detection
    """
    def __init__(self, cfg, in_channels):
        super(VP_MonoFM_Head, self).__init__()
        
        self.cfg = cfg
        self.predictor = make_predictor(cfg, in_channels)
        self.loss_evaluator = make_loss_evaluator(cfg)
        self.post_processor = make_post_processor(cfg)

    def forward(self, features, targets=None, test=False):
        x = self.predictor(features, targets)

        if self.training:
            loss_dict, log_loss_dict = self.loss_evaluator(x, targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, visualize_preds = self.post_processor(
                x, targets, test=test, features=features
            )
            return result, eval_utils, visualize_preds


def build_vp_monofm_head(cfg, in_channels):
    """Build VP-MonoFM detection head"""
    return VP_MonoFM_Head(cfg, in_channels)


class VP_MonoFM_Detector(nn.Module):
    '''Visual Prompt-guided Monocular 3D Object Detection with Multiscale Fusion
    
    Main parts:
    - backbone: DLA feature extractor
    - head: VP-MonoFM detection head (MDF + 3DR + VPF + 2D detection)
    '''

    def __init__(self, cfg):
        super(VP_MonoFM_Detector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = build_vp_monofm_head(cfg, self.backbone.out_channels)
        self.test = cfg.DATASETS.TEST_SPLIT == 'test'

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images = to_image_list(images)
        features = self.backbone(images.tensors)

        if self.training:
            loss_dict, log_loss_dict = self.heads(features, targets)
            return loss_dict, log_loss_dict
        else:
            result, eval_utils, visualize_preds = self.heads(features, targets, test=self.test)
            return result, eval_utils, visualize_preds