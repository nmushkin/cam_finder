# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import torch
# import torchvision
# import matplotlib.pyplot as plt

from detection_models import get_fasterrcnn_model
from voc_coco_dataset import VocXmlDataset
import torchvision_scripts.transforms as T
from torchvision_scripts import utils
from torchvision_scripts.engine import train_one_epoch, evaluate


def train_model():
    IMAGE_DIR = './data/images/new_cameras/'
    LABEL_DIR = './data/new_cameras_labels/'

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Not including background - model adds that
    num_classes = 2
    class_names = ['fixed_cam', 'round_cam']
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
    model = get_fasterrcnn_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


if __name__ == "__main__":
    train_model()
