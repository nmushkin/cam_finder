# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, \
                                             RegionProposalNetwork, RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


def get_fasterrcnn_model(num_classes, feature_extract_only=False):

    model = fasterrcnn_resnet50_fpn(pretrained=True)
    if feature_extract_only:
        set_grad_required(model, not feature_extract_only)

    anchor_sizes = ((16,), (32,), (64,), (128,), (256,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_gen = AnchorGenerator(sizes=anchor_sizes,
                                 aspect_ratios=aspect_ratios)
    print(anchor_gen.num_anchors_per_location()[0])
    rpn_head = RPNHead(256, anchor_gen.num_anchors_per_location()[0])

    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    model.rpn = RegionProposalNetwork(
        anchor_generator=anchor_gen,
        head=rpn_head,
        fg_iou_thresh=0.8,
        bg_iou_thresh=0.2,
        batch_size_per_image=256,
        positive_fraction=0.5,
        pre_nms_top_n=rpn_pre_nms_top_n,
        post_nms_top_n=rpn_post_nms_top_n,
        nms_thresh=0.7
    )

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes + 1  # +1 for background class
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
