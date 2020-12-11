import os
import xml.etree.ElementTree as etree

import torch
from torch.utils.data.dataset import Dataset
from PIL import Image


class VocXmlDataset(Dataset):
    """A Pytorch Dataset for using images and annotations in VOC format"""

    def __init__(self, image_dir, xml_dir, class_names, im_length, transforms=None):
        self.image_dir = image_dir
        self.xml_dir = xml_dir
        self.xml_files = sorted(os.listdir(xml_dir))
        self.class_names = {
            class_names[n]: n + 1 for n in range(len(class_names))
            }
        self.transforms = transforms
        self.im_length = im_length

    def __getitem__(self, idx):
        xml_path = os.path.join(self.xml_dir, self.xml_files[idx])
        sample = voc_xml_to_dict(xml_path)
        image_path = os.path.join(self.image_dir, sample['image'])
        image = Image.open(image_path).convert('RGB')
        boxes = sample['boxes']

        if self.im_length is not None:
            if image.height > image.width:
                old_length = image.height
            else:
                old_length = image.width
            ratio = self.im_length / old_length
            new_size = (round(image.width * ratio), round(image.height * ratio))

            for box in range(len(boxes)):
                boxes[box] = resize_bbox(
                    image.height,
                    new_size[1],
                    image.width,
                    new_size[0],
                    boxes[box]
                    )
            image = image.resize((new_size[0], new_size[1]))
        areas = []
        for box in boxes:
            areas.append((box[2] - box[0]) * (box[3] - box[1]))
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(
            [self.class_names[c] for c in sample['labels']],
            dtype=torch.int64)
        image_id = torch.as_tensor([idx], dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        is_crowd = torch.zeros((len(sample['boxes']),), dtype=torch.int64)

        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = image_id
        target['area'] = areas
        target['iscrowd'] = is_crowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.xml_files)


def voc_xml_to_dict(xml_path):
    """Parses voc-format xml file, like those made by labelImg"""

    xml = etree.parse(xml_path)
    root = xml.getroot()
    image_name = root.find('filename').text
    labels = []
    boxes = []

    for obj in root.findall('object'):
        labels.append(obj.find('name').text)
        bbox = obj.find('bndbox')
        bounds = [
            float(bbox.find('xmin').text),
            float(bbox.find('ymin').text),
            float(bbox.find('xmax').text),
            float(bbox.find('ymax').text)
        ]
        boxes.append(bounds)

    return {
        'image': image_name,
        'labels': labels,
        'boxes': boxes,
    }


def resize_bbox(h1, h2, w1, w2, bbox):
    """Resizes a bounding box given old and new image dimensions"""

    hr = h2 / h1
    wr = w2 / w1
    x1, x2 = bbox[0] * wr, bbox[2] * wr
    y1, y2 = bbox[1] * hr, bbox[3] * hr
    return [x1, y1, x2, y2]
