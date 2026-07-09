
# Class weights for handling imbalanced dataset
# Generated automatically - add to loss.py

import torch

CLASS_WEIGHTS = torch.tensor([
    0.129540,
    0.777659,
    0.246601,
    1.799968,
    0.484077,
    1.714683,
    0.501042,
    1.799968,
    1.799968,
    1.539740,
    0.711346,
    1.056348,
    1.799968,
    0.703179,
    0.476636,
    0.295151,
    0.737040,
    1.799968,
    0.266018,
    0.904484,
    0.132945,
    0.734093,
    0.679781,
    0.130517,
    0.292436,
    1.288261,
    1.799968,
    0.379835,
    1.799968,
    0.411493,
    0.982351,
    0.142514,
    1.799968,
    1.799968,
    0.066175,
    0.203344,
    1.799968,
    1.261423,
    1.799968,
    1.799968,
    0.551731,
    1.799968,
    1.799968
], dtype=torch.float32)

# Use in loss function:
# cls_loss = F.binary_cross_entropy_with_logits(
#     cls_pred, cls_target, 
#     weight=CLASS_WEIGHTS[target_classes].to(device)
# )
