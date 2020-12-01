# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_fasterrcnn_model(num_classes, feature_extract_only=False):
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

    return model


def set_grad_required(model, grad_required):
    for param in model.parameters():
        param.requires_grad = grad_required
