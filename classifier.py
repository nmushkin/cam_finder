import os

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import numpy as np

from detection_models import get_fasterrcnn_model


def classify(model, image_folder, save_folder):
    """Classifies objects in images from image_folder

       Saves annotated images in save_folder"""
    model.eval()
    for fname in os.listdir(image_folder):
        if 'jpeg' not in fname:
            continue
        prefix = fname[:fname.index('.jpeg')]
        im = Image.open(os.path.join(image_folder, fname))
        rgb_tiles = [F.to_tensor(arr) for arr in rgb_quarter(im)]
        predictions = model(rgb_tiles)
        for tile_no, tile_preds in enumerate(predictions):
            save = False
            image = torchvision.transforms.ToPILImage()(rgb_tiles[tile_no])
            image = image.convert('RGB')
            box_im = image.copy()
            for cam_no in range(len(tile_preds['boxes'])):
                probability = tile_preds['scores'][cam_no].item()
                if probability >= .8:
                    save = True
                    print(probability, prefix)
                    boxes = tile_preds['boxes'][cam_no].tolist()
                    box_im = draw_bbox(box_im, boxes, probability)
            if save:
                fname = f'{prefix}-{tile_no}'
                image.save(fp=(f'{os.path.join(save_folder, fname)}.jpeg'))
                box_fname = f'{fname}-box'
                box_im.save(
                    fp=(f'{os.path.join(save_folder, box_fname)}.jpeg')
                )


def rgb_quarter(image):
    """Splits a PIL image into quarters each 800x800

       Returns a list of numpy arrays representing each quarter"""

    img = image.resize((1600, 1600))
    img_rgb = np.array(img.convert('RGB'))
    y_num = img_rgb.shape[1] // 2
    x_num = img_rgb.shape[0] // 2
    indices = [(x, y) for x in range(0, 2) for y in range(0, 2)]
    rgb_tiles = [
        img_rgb[x*x_num: (x+1)*x_num, y*y_num: (y+1)*y_num] for x, y in indices
    ]
    return rgb_tiles


def draw_bbox(img, bbox, prob):
    """Draws a bounding box and probability in an image"""

    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(3, 252, 57))
    draw.text(xy=bbox[0:2], text=str(int(prob*100)))
    return img


weights = './data/models/resnet_50_b2_10e.pth'
image_folder = './data/walk_pics'
to_folder = './data/images/new_cameras_predictions'
model = get_fasterrcnn_model(2, False)
state = torch.load(weights)
model.load_state_dict(state)
model.to(torch.device('cpu'))
classify(model, image_folder, to_folder)
