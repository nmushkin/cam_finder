import os
import json

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import numpy as np

from detection_models import get_fasterrcnn_model


def classify(model, image_folder, save_folder, json_save_file, device=torch.device('cuda')):
    """Classifies objects in images from image_folder

       Saves annotated images in save_folder as well as info in json_save_file"""

    model.eval()
    cam_info = {'locations': {}}

    for file_num, fname in enumerate(os.listdir(image_folder)):
        if 'jpeg' not in fname:
            continue
        prefix = fname[:fname.index('.jpeg')]
        im = Image.open(os.path.join(image_folder, fname))
        # rgb_tiles = [F.to_tensor(arr).to(device) for arr in image_split(im)]
        rgb_tiles = [F.to_tensor(im)]
        predictions = model(rgb_tiles)
        location = prefix.split('_')
        current_json = {'lat': float(location[0]), 'lng': float(location[1]),
                        'heading': float(location[2]), 'tiles': []}
        for tile_num, tile_preds in enumerate(predictions):
            tile_image = torchvision.transforms.ToPILImage()(rgb_tiles[tile_num])
            im, box_im, tile_info = get_tile_info(tile_num, tile_preds, tile_image)
            if tile_info['cameras']:
                fname = f'{prefix}-{tile_num}'
                # im.save(fp=(f'{os.path.join(save_folder, fname)}.jpeg'),
                #         quality=95, subsampling=0)
                box_fname = f'{fname}-box'
                box_im.save(
                    fp=(f'{os.path.join(save_folder, box_fname)}.jpeg')
                )
                tile_info['image'] = f'{box_fname}.jpeg'
                current_json['tiles'].append(tile_info)

        if current_json['tiles']:
            loc_hash = f'{current_json["lat"]}_{current_json["lng"]}'
            if loc_hash not in cam_info['locations']:
                cam_info['locations'][loc_hash] = {
                    'lat': current_json['lat'],
                    'lng': current_json['lng'],
                    'headings': {}
                }
            cam_info['locations'][loc_hash]['headings'][current_json['heading']] = current_json['tiles']

        if file_num % 10 == 0:
            with open(json_save_file, 'w') as json_f:
                json.dump(cam_info, json_f)

    with open(json_save_file, 'w') as json_f:
        json.dump(cam_info, json_f)


def get_tile_info(tile_num, tile_preds, tile_image):
    im = tile_image.convert('RGB')
    box_im = im.copy()
    tile_info = {'cameras': []}
    for cam_no in range(len(tile_preds['boxes'])):
        probability = tile_preds['scores'][cam_no].item()
        if probability >= .2:
            label = tile_preds['labels'][cam_no].item()
            boxes = tile_preds['boxes'][cam_no].tolist()
            box_im = draw_bbox(box_im, boxes, probability, label)
            tile_info['cameras'].append([label, probability, boxes])

    return im, box_im, tile_info


def image_split(image):
    """Splits a PIL image into tiles each 800x800

       Returns a list of numpy arrays representing each quarter"""

    img = image.resize((2400, 2400))
    img_rgb = np.array(img.convert('RGB'))
    y_num = img_rgb.shape[1] // 3
    x_num = img_rgb.shape[0] // 3
    indices = [(x, y) for x in range(0, 3) for y in range(0, 3)]
    rgb_tiles = [
        img_rgb[x*x_num: (x+1)*x_num, y*y_num: (y+1)*y_num] for x, y in indices
    ]
    return rgb_tiles


def draw_bbox(img, bbox, prob, label):
    """Draws a bounding box and probability in an image"""

    draw = ImageDraw.Draw(img)
    draw.rectangle(bbox, outline=(3, 252, 57))
    draw.text(xy=bbox[0:2], text=str(int(prob*100)))
    draw.text(xy=[bbox[0], bbox[1]+10], text=str(label))
    return img


weights = 'resnet_50_05_06_2021.pth'
image_folder = './data/walk_test'
to_folder = './data/walk_test'
json_save_file = './data/test_json.json'
model = get_fasterrcnn_model(2, False)
state = torch.load(weights)
model.load_state_dict(state)
model.to(torch.device('cpu'))

classify(model, image_folder, to_folder, json_save_file, torch.device('cpu'))
