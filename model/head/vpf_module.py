import torch
import torch.nn as nn
import torch.nn.functional as F


class VPF(nn.Module):
    """Visual Prompt Feature Module
    
    Enhances object features using visual prompt guidance.
    Creates attention-based feature enhancement with learnable sigmoid-based weighting.
    """
    def __init__(self, in_channels):
        super(VPF, self).__init__()
        
        # Dual ConvBN-SiLU paths with skip connection
        self.path1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),
        )
        
        # Conv-Sigmoid layer for attention map
        self.attention_layer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
        # Learnable parameters for sigmoid weighting
        # Formula (10): wi = 1 / (1 + exp((β*si - b) / T))
        self.register_parameter('beta', nn.Parameter(torch.tensor(1.0)))
        self.register_parameter('bias', nn.Parameter(torch.tensor(0.5)))
        self.register_parameter('temperature', nn.Parameter(torch.tensor(1.0)))

    def forward(self, features, bboxes_2d=None, object_sizes=None, training=True):
        """
        Args:
            features: input features (B, C, H, W)
            bboxes_2d: 2D bounding boxes from GT (training) or detector (testing)
                      shape: (N, 4) where N is number of objects
            object_sizes: object sizes/area for weighting (N,)
            training: whether in training mode
        
        Returns:
            enhanced_features: Fvp (B, C, H, W)
            attention_map: F' (B, 1, H, W)
        """
        B, C, H, W = features.shape
        
        # Dual path processing with skip connection
        x1 = self.path1(features)
        x2 = self.path2(features)
        x = x1 + x2 + features  # Skip connection
        
        # Generate attention map F'
        attention_map = self.attention_layer(x)  # (B, 1, H, W)
        
        if training and bboxes_2d is not None and object_sizes is not None:
            # Training mode: use visual prompt mask with GT information
            visual_prompt = self._create_visual_prompt(
                bboxes_2d, object_sizes, B, H, W, features.device
            )
            
            # Multiply attention map with visual prompt
            # VF' = F' * VP
            adjusted_attention = attention_map * visual_prompt  # (B, 1, H, W)
        else:
            # Testing mode: use attention map directly as prompt
            adjusted_attention = attention_map
        
        # Multiply with original features
        # Fvp = VF' * F
        enhanced_features = adjusted_attention * features  # (B, C, H, W)
        
        return enhanced_features, adjusted_attention

    def _create_visual_prompt(self, bboxes_2d, object_sizes, batch_size, feat_h, feat_w, device):
        """
        Create visual prompt mask from 2D bounding boxes and object sizes.
        
        Formula (10): wi = 1 / (1 + exp((β*si - b) / T))
        
        Args:
            bboxes_2d: (N, 4) - [x1, y1, x2, y2] in image coordinates
            object_sizes: (N,) - object sizes/areas
            batch_size: B
            feat_h, feat_w: feature map dimensions
            
        Returns:
            visual_prompt: (B, 1, H, W) - prompt mask
        """
        # Initialize mask with ones (preserve background)
        Wm = torch.ones(batch_size, feat_h, feat_w, device=device)
        
        if bboxes_2d is not None and len(bboxes_2d) > 0:
            # Calculate weights using learnable sigmoid-based formula
            # Formula (10): wi = 1 / (1 + exp((β*si - b) / T))
            object_sizes = torch.clamp(object_sizes, min=1e-6)
            beta = torch.clamp(self.beta, min=0.1)
            temp = torch.clamp(self.temperature, min=0.1)
            
            weights = 1.0 / (1.0 + torch.exp((beta * object_sizes - self.bias) / temp))
            weights = torch.clamp(weights, min=0.0, max=1.0)
            
            # Assign (1 + wi) to each bbox area
            for idx, bbox in enumerate(bboxes_2d):
                x1, y1, x2, y2 = bbox.int()
                # Clamp to feature map bounds
                x1 = max(0, min(x1, feat_w - 1))
                y1 = max(0, min(y1, feat_h - 1))
                x2 = max(x1 + 1, min(x2, feat_w))
                y2 = max(y1 + 1, min(y2, feat_h))
                
                if x2 > x1 and y2 > y1:
                    # Assign 1 + wi to bbox area, keep maximum for overlapping areas
                    weight_value = 1.0 + weights[idx].item()
                    Wm[:, y1:y2, x1:x2] = torch.max(
                        Wm[:, y1:y2, x1:x2],
                        torch.tensor(weight_value, device=device)
                    )
        
        # Perform maximum pooling to match feature dimensions
        # This is a simplification - in practice, use proper downsampling
        if Wm.shape[1] > feat_h or Wm.shape[2] > feat_w:
            Wm = F.adaptive_max_pool2d(
                Wm.unsqueeze(1),
                (feat_h, feat_w)
            ).squeeze(1)
        
        visual_prompt = Wm.unsqueeze(1)  # (B, 1, H, W)
        return visual_prompt
        
    def forward(self, features, targets=None, test=False):
        """
        Args:
            features: Feature map from backbone (B, C, H, W)
            targets: Ground truth annotations during training
            test: Boolean flag for test mode
            
        Returns:
            Fvp: Enhanced feature map (B, C, H, W)
            visual_prompt: Visual prompt mask used (for visualization/debugging)
        """
        batch_size, channels, H_feat, W_feat = features.shape
        
        # Step 1: Generate attention map F' using dual Conv-BN-SiLU paths
        # First path
        f1 = self.conv1_1(features)
        f1 = self.bn1_1(f1)
        f1 = self.silu1(f1)
        
        # Second path
        f2 = self.conv1_2(features)
        f2 = self.bn1_2(f2)
        f2 = self.silu2(f2)
        
        # Combine with skip connection (add with input for residual connection)
        f_combined = f1 + f2 + features
        
        # Attention map with sigmoid
        F_prime = self.conv_attention(f_combined)  # (B, C, H, W)
        
        # Step 2: Create visual prompt mask during training
        if self.training and targets is not None:
            # Get original image dimensions
            img_height = targets.get('image_height', H_feat * 2) if isinstance(targets, dict) else H_feat * 2
            img_width = targets.get('image_width', W_feat * 2) if isinstance(targets, dict) else W_feat * 2
            
            # Get 2D bounding boxes and object sizes from GT
            bboxes_2d = targets.get('bboxes_2d', None) if isinstance(targets, dict) else None
            
            if bboxes_2d is not None:
                # Create mask at image resolution
                Wm = self._create_visual_prompt_mask(
                    bboxes_2d,
                    img_height, img_width,
                    batch_size,
                    device=features.device
                )
                
                # Downsample mask to feature map size using max pooling
                # Maintain consistency with feature map size
                scale_h = img_height / H_feat
                scale_w = img_width / W_feat
                pool_size = max(int(scale_h), int(scale_w))
                
                if pool_size > 1:
                    VP = F.adaptive_max_pool2d(Wm, (H_feat, W_feat))
                else:
                    VP = Wm
            else:
                # No bounding boxes: use uniform mask
                VP = torch.ones(batch_size, 1, H_feat, W_feat, device=features.device)
            
            # Multiply attention map with visual prompt
            VF_prime = F_prime * VP  # Element-wise multiplication
        else:
            # During testing: no visual prompt, attention map directly multiplied with features
            VP = None
            VF_prime = F_prime
        
        # Step 3: Enhance feature with adjusted attention
        # Fvp = VF' ⊗ F (element-wise multiplication along channels)
        Fvp = VF_prime * features
        
        return Fvp, VP
    
    def _create_visual_prompt_mask(self, bboxes_2d, img_height, img_width, batch_size, device):
        """
        Create visual prompt mask from 2D bounding boxes
        
        Formula (10): wi = 1 / (1 + exp((β·si - b) / T))
        where si is the size of i-th object
        
        Mask assignment:
        - Area covered by object i: 1 + wi
        - Area not covered: 1
        - Overlapping areas: max(wi)
        
        Args:
            bboxes_2d: 2D bounding boxes (B, N, 4) format: (x1, y1, x2, y2)
            img_height: Image height
            img_width: Image width
            batch_size: Batch size
            device: Device for tensor creation
            
        Returns:
            Wm: Visual prompt mask (B, 1, img_height, img_width)
        """
        # Initialize mask to 1 (preserve attention values)
        Wm = torch.ones(batch_size, 1, img_height, img_width, device=device)
        
        if bboxes_2d is None:
            return Wm
        
        # Process each image in batch
        for b in range(batch_size):
            boxes = bboxes_2d[b] if len(bboxes_2d.shape) == 3 else bboxes_2d  # (N, 4)
            
            for box_idx, box in enumerate(boxes):
                if box is None or torch.all(box == 0):
                    continue
                
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Clamp to image boundaries
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(x1 + 1, min(x2, img_width))
                y2 = max(y1 + 1, min(y2, img_height))
                
                # Calculate object size (bounding box area)
                size = (x2 - x1) * (y2 - y1)
                
                # Calculate weight using sigmoid formula (formula 10)
                # wi = 1 / (1 + exp((β·si - b) / T))
                weight = 1.0 / (1.0 + torch.exp((self.beta * size - self.b) / self.T))
                weight = weight.item() if isinstance(weight, torch.Tensor) else weight
                
                # Assign 1 + wi to object area, keep overlapping max
                Wm[b, 0, y1:y2, x1:x2] = torch.max(
                    Wm[b, 0, y1:y2, x1:x2],
                    torch.tensor(1.0 + weight, device=device)
                )
        
        return Wm
