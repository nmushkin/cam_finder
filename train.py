# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
from math import isfinite

import torch
import PIL
from PIL import ImageDraw
import torchvision

import matplotlib.pyplot as plt

from detection_models import get_fasterrcnn_model, set_grad_required, get_model
from voc_coco_dataset import VocXmlDataset
import torchvision_scripts.transforms as T
from torchvision_scripts import utils
from torchvision_scripts.engine import evaluate

IMAGE_DIR = './data/images/new_cameras/'
LABEL_DIR = './data/all_round_labels/'


def train_model(class_names, model, feature_extract_only=True, epochs=10):
    torch.manual_seed(4)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # use our dataset and defined transformations
    dataset = VocXmlDataset(
        IMAGE_DIR, LABEL_DIR, class_names, None, get_transform(train=True)
        )
    dataset_test = VocXmlDataset(
        IMAGE_DIR, LABEL_DIR, class_names, None, get_transform(train=False)
    )
    # im, target = dataset.__getitem__(169)
    # print(target)
    # plt.imshow(torchvision.transforms.ToPILImage()(im))
    # plt.show()
    # return
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    print(f'{len(params)} Params To Train')
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.7, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=4,
                                                   gamma=0.5)

    for epoch in range(epochs):
        print(f'Learning rate is {lr_scheduler.get_last_lr}')
        # train for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    model.cpu()
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    print(f'Epoch {epoch}')
    model.train()
    lowest_loss = None
    last_percent = -1
    total_loss = 0
    num_batches = len(data_loader)

    for batch, (images, targets) in enumerate(data_loader):
        percent = round(batch / num_batches * 100)
        if percent != last_percent:
            print(f'{percent}%', end=' ', flush=True)
            last_percent = percent

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        if not isfinite(losses):
            print(f'Loss is {losses}, stopping training')
            print(loss_dict)
            return

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

    total_loss = total_loss / num_batches
    print(f'Total Loss: {total_loss}')

    if lowest_loss is None or total_loss < lowest_loss:
        model.cpu()
        torch.save(model.state_dict(), './resnet_50_1500_nofeature.pth')
        model.to(device)
        lowest_loss = total_loss


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def draw_bbox(img, bbox, text):
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(3, 252, 57))
    draw.text(xy=bbox[0:2], text=str(int(text*100)))
    return img


if __name__ == "__main__":
    classes = ['fixed_cam', 'round_cam']
    num_classes = len(classes)
    # print('Feature Extracting')
    model = get_fasterrcnn_model(num_classes, False)
    # model = get_model(num_classes, False)
    # print('Training On All Params')
    # set_grad_required(model, True)
    # train_model saves best epoch weights
    model = train_model(epochs=20, class_names=classes, model=model)
    # torch.save(model.state_dict(), './data/resnet_50_1500_20e_nofeature.pth')
