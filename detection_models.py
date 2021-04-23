# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

import torch


def get_fasterrcnn_model(num_classes, feature_extract_only=False):

    anchor_gen = AnchorGenerator(sizes=((16, 32, 64, 128, 256), ))
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    if feature_extract_only:
        set_grad_required(model, not feature_extract_only)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes + 1  # +1 for background class
    )

    rpn_head = RPNHead(256, anchor_gen.num_anchors_per_location()[0])
    # RPN parameters

    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    model.rpn = RegionProposalNetwork(
        anchor_gen, rpn_head,
        0.7, 0.3,
        256, 0.5,
        rpn_pre_nms_top_n, rpn_post_nms_top_n, 0.7,
        score_thresh=0.0
    )

    return model


def set_grad_required(model, grad_required):
    for param in model.parameters():
        param.requires_grad = grad_required


def get_model(num_classes, features_only=False):
    backbone = resnet_fpn_backbone('resnet18', True, trainable_layers=1)
    model = FasterRCNN(backbone, num_classes)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes + 1  # +1 for background class
    )
    print(model)
    return model
