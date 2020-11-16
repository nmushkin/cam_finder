# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
from math import isfinite

import torch
import PIL
from PIL import ImageDraw
import torchvision

from detection_models import get_fasterrcnn_model
from voc_coco_dataset import VocXmlDataset
import torchvision_scripts.transforms as T
from torchvision_scripts import utils
from torchvision_scripts.engine import evaluate

IMAGE_DIR = './data/images/new_cameras/'
LABEL_DIR = './data/new_cameras_labels/'


def train_model(class_names, feature_extract_only=True, epochs=10):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # Not including background - model adds that
    num_classes = len(class_names)
    # use our dataset and defined transformations
    dataset = VocXmlDataset(
        IMAGE_DIR, LABEL_DIR, class_names, (600, 600), get_transform(train=True)
        )
    dataset_test = VocXmlDataset(
        IMAGE_DIR, LABEL_DIR, class_names, (600, 600), get_transform(train=False)
    )
    # im, target = dataset.__getitem__(100)
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
        dataset, batch_size=5, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_fasterrcnn_model(num_classes, feature_extract_only)
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    print(params)
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    model.cpu()
    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    print(f'Epoch {epoch}')

    for batch, (images, targets) in enumerate(data_loader):
        if batch % 5 == 0:
            percent = round(batch / len(data_loader) * 100)
            print(f'{percent}%', end=' ', flush=True)
        # im = torchvision.transforms.ToPILImage()(images[0])
        # box = targets[0]['boxes'][0].tolist()
        # draw_bbox(im, box, 0.1).show()
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        if not isfinite(losses):
            print(f'Loss is {losses}, stopping training')
            print(loss_dict)
            return

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def draw_bbox(img, bbox, prob):
    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(3, 252, 57))
    draw.text(xy=bbox[0:2], text=str(int(prob*100)))
    return img


if __name__ == "__main__":
    classes = ['fixed_cam', 'round_cam']
    model = train_model(epochs=10, class_names=classes)
    torch.save(model.state_dict(), './data/models/resnet_50.pth')
