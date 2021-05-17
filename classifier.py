import os
import json

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw
import numpy as np

from detection_models import get_fasterrcnn_model

IMAGE_SIZE = 2400
NUM_TILES = 3
TILE_SIZE = IMAGE_SIZE / NUM_TILES


def classify(model, image_folder, save_folder, json_save_file, device=torch.device('cuda')):
    """Classifies objects in images from image_folder

       Saves annotated images in save_folder as well as info in json_save_file"""

    model.to(device)
    model.eval()
    cam_info = {'locations': {}}

    for file_num, fname in enumerate(os.listdir(image_folder)):
        if 'jpeg' not in fname:
            continue
        prefix = fname[:fname.index('.jpeg')]
        im = Image.open(os.path.join(image_folder, fname))
        rgb_tiles = [F.to_tensor(im).to(device)]
        prediction = model(rgb_tiles)[0]
        location = prefix.split('_')
        print(prediction)
        im, box_im, tile_info = get_prediction_info(prediction, im)
        if tile_info['cameras']:
            box_fname = f'{prefix}-box'
            box_im.save(
                fp=(f'{os.path.join(save_folder, box_fname)}.jpeg')
            )

            for tile_no, tile_image in enumerate(get_occupied_tile_images(
                                                  prediction['boxes'],
                                                  im,
                                                  IMAGE_SIZE, TILE_SIZE)):
                fname = f'{prefix}-{tile_no}'
                tile_image = torchvision.transforms.ToPILImage()(tile_image)
                tile_image.save(fp=(f'{os.path.join(save_folder, fname)}.jpeg'),
                                quality=95, subsampling=0)

            tile_info['image'] = f'{box_fname}.jpeg'
            current_json = {
                'lat': float(location[0]), 'lng': float(location[1]),
                'heading': float(location[2]), 'tiles': [tile_info]}
            loc_hash = f'{current_json["lat"]}_{current_json["lng"]}'
            if loc_hash not in cam_info['locations']:
                cam_info['locations'][loc_hash] = {
                    'lat': current_json['lat'],
                    'lng': current_json['lng'],
                    'headings': {}
                }
            cam_info['locations'][loc_hash]['headings'][current_json['heading']] = current_json['tiles']

        if file_num % 50 == 0:
            with open(json_save_file, 'w') as json_f:
                json.dump(cam_info, json_f)

    with open(json_save_file, 'w') as json_f:
        json.dump(cam_info, json_f)


def get_occupied_tile_images(bbox_list, image, image_size, tile_size):
    tiles_per_dim = image_size // tile_size
    occupied_tiles = set()
    for box_no in range(len(bbox_list)):
        bbox = bbox_list[box_no].tolist()
        # bottom_left = (bbox[0], bbox[1])
        # bottom_right = (bbox[2], bbox[1])
        # top_left = (bbox[0], bbox[3])
        # top_right = (bbox[2], bbox[3])
        for x in [bbox[0], bbox[2]]:
            for y in [bbox[1], bbox[3]]:
                occupied_tiles.add(get_tile_from_point(x, y, tile_size, tiles_per_dim))
    print(occupied_tiles)

    image_tiles = image_split(image)
    return [tile[1] for tile in enumerate(image_tiles) if tile[0] in occupied_tiles]


def get_tile_from_point(x, y, tile_width, tiles_per_dim):
    x_tile = x // tile_width
    y_tile = y // tile_width
    return y_tile * tiles_per_dim + x_tile


def get_prediction_info(prediction, image):
    im = image.convert('RGB')
    box_im = im.copy()
    tile_info = {'cameras': []}
    for cam_no in range(len(prediction['boxes'])):
        probability = prediction['scores'][cam_no].item()
        if probability >= .01:
            label = prediction['labels'][cam_no].item()
            bbox = prediction['boxes'][cam_no].tolist()
            box_im = draw_bbox(box_im, bbox, probability, label)
            tile_info['cameras'].append([label, probability, bbox])

    return im, box_im, tile_info


def image_split(image):
    """Splits a PIL image into tiles each 800x800

       Returns a list of numpy arrays representing each tile"""

    img = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_rgb = np.array(img.convert('RGB'))
    y_num = img_rgb.shape[1] // NUM_TILES
    x_num = img_rgb.shape[0] // NUM_TILES
    indices = [(x, y) for x in range(0, NUM_TILES) for y in range(0, NUM_TILES)]
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
image_folder = './data/images/walk_pics/dt_atlanta_chunk_1'
to_folder = './data/walk_test'
json_save_file = './data/test_json.json'
model = get_fasterrcnn_model(2, False)
state = torch.load(weights)
model.load_state_dict(state)

classify(model, image_folder, to_folder, json_save_file, torch.device('cpu'))
