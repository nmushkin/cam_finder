import os
import json
import xml.etree.ElementTree as etree
import math

from PIL import Image


def png_to_jpeg(xml_label_folder, image_folder):
    """Converts images and their mentions in relevant xml to .jpeg"""
    xml_files = os.listdir(xml_label_folder)
    for file_name in xml_files:
        xml_path = os.path.join(xml_label_folder, file_name)
        xml = etree.parse(xml_path)
        root = xml.getroot()
        image_name = root.find('filename').text
        if 'png' in image_name:
            im_path = os.path.join(image_folder, image_name)
            if not os.path.isfile(im_path):
                print(im_path)
                continue
            im = Image.open(im_path).convert('RGB')
            im.save(im_path.replace('png', 'jpeg'))
            image_path = root.find('path').text
            new_name = image_name.replace('.png', '.jpeg')
            root.find('filename').text = new_name
            root.find('path').text = image_path.replace('.png', '.jpeg')
            os.remove(im_path)
            xml.write(xml_path)


# def round_cam_labels(xml_label_folder, new_xml_label_folder):
#     """Changes labels in VOC xml"""
#     xml_files = os.listdir(xml_label_folder)
#     for file_name in xml_files:
#         xml_path = os.path.join(xml_label_folder, file_name)
#         xml = etree.parse(xml_path)
#         root = xml.getroot()
#         for obj in root.findall('object'):
#             if obj.find('name').text == 'disc_cam':
#                 obj.find('name').text = 'round_cam'
#         new_path = os.path.join(new_xml_label_folder, file_name)
#         xml.write(new_path)

def increase_box_size(xml_label_folder, new_xml_folder, pct_increase):
    """Increases bbox size in VOC xml"""
    xml_files = os.listdir(xml_label_folder)
    for file_name in xml_files:
        xml_path = os.path.join(xml_label_folder, file_name)
        xml = etree.parse(xml_path)
        root = xml.getroot()
        size = root.find('size')
        im_width = int(size.find('width').text)
        im_height = int(size.find('height').text)

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            width_add = math.floor((xmax - xmin) * pct_increase / 100)
            height_add = math.floor((ymax - ymin) * pct_increase / 100)

            bbox.find('xmin').text = str(max(0, xmin - width_add))
            bbox.find('ymin').text = str(max(0, ymin - height_add))
            bbox.find('xmax').text = str(min(im_width - 1, xmax + width_add))
            bbox.find('ymax').text = str(min(im_height - 1, ymax + height_add))

        new_path = os.path.join(new_xml_folder, file_name)
        xml.write(new_path)


def bbox_size_stats(xml_label_folder):
    """Increases bbox size in VOC xml"""
    xml_files = os.listdir(xml_label_folder)
    heights = []
    widths = []
    ratios = []
    for file_name in xml_files:
        xml_path = os.path.join(xml_label_folder, file_name)
        xml = etree.parse(xml_path)
        root = xml.getroot()

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            widths.append(xmax - xmin)
            heights.append(ymax - ymin)
            ratios.append((ymax - ymin) / (xmax - xmin))

    heights.sort()
    widths.sort()
    print(sum(heights) / len(heights))
    print(sum(widths) / len(widths))
    print(sum(ratios) / len(ratios))
    print(min(heights))
    print(min(widths))
    print(max(heights))
    print(max(widths))
    print(heights)
    print(widths)


def round_cam_labels(xml_label_folder):
    """Changes labels in VOC xml"""
    xml_files = os.listdir(xml_label_folder)
    for file_name in xml_files:
        xml_path = os.path.join(xml_label_folder, file_name)
        try:
            xml = etree.parse(xml_path)
            root = xml.getroot()
            if len(root.findall('object')) == 0:
                print('hey')
                print(xml_path)
        except Exception:
            print(file_name)
        # new_path = os.path.join(new_xml_label_folder, file_name)
        # xml.write(new_path)


def delete_unused_images(xml_dir, im_dir):
    photo_list = set(os.listdir(im_dir))
    xml_list = set(os.listdir(xml_dir))
    for f in photo_list:
        if 'jp' in f:
            prefix = f[:f.index('.jp')]
        elif 'png' in f:
            prefix = f[:f.index('.png')]
        xml_file = f'{prefix}.xml'
        if xml_file not in xml_list:
            print(f)
            os.remove(os.path.join(im_dir, f))


def delete_unused_xml(xml_dir, im_dir):
    photo_list = set(os.listdir(im_dir))
    xml_list = set(os.listdir(xml_dir))
    for f in xml_list:
        prefix = f[:f.index('.xml')]
        jpg_file = f'{prefix}.jpg'
        jpeg_file = f'{prefix}.jpeg'
        if jpeg_file not in photo_list and jpg_file not in photo_list:
            print(f)
            os.rename(
                    os.path.join(xml_dir, f), 
                    os.path.join('/Users/noahmushkin/codes/cam_finder/classification/data/images/old_cameras', f)
                )
            # os.remove(os.path.join(xml_dir, f))


def image_size_check(im_dir, move_dir):
    photo_list = set(os.listdir(im_dir))
    count = 0
    for f in photo_list:
        if 'DS' in f:
            continue
        im = Image.open(os.path.join(im_dir, f))
        if im.size[0] != 800:
            # if move_dir is not None:
            #     os.rename(
            #         os.path.join(im_dir, f), 
            #         os.path.join(move_dir, f)
            #     )
            count += 1
            print(im.size)
    print(count)


def threshhold():
    image_set = set()
    with open('/Users/noahmushkin/Downloads/cambridge-1.json') as f:
        cam_file = json.load(f)
    locations = cam_file['locations'].values()
    for headings in [h['headings'] for h in locations]:
        for heading in headings.values():
            for tile in heading:
                for cam in tile['cameras']:
                    if cam[1] >= .95:
                        image_set.add(tile['image'])
    print(len(image_set))
    for im in image_set:
        im_n = im.replace('-box', '')
        os.rename(os.path.join('/Users/noahmushkin/Downloads/ProcessedPics 4', im_n),
                  os.path.join('/Users/noahmushkin/Downloads/chosenpics', im_n))
        # try:
        #     Image.open(os.path.join('/Users/noahmushkin/Downloads/ProcessedPics 4', im)).show()
        # except Exception:
        #     pass


def image_filter(reference_dir, source_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    name_set = set()
    for f in set(os.listdir(reference_dir)):
        if 'jpeg' not in f:
            continue
        base_name = f[:f.index('-box')]
        name_set.add(base_name)
    for f in set(os.listdir(source_dir)):
        if 'jpeg' not in f:
            continue
        base_name = f[:f.index('-')]
        if base_name in name_set:
            old_path = os.path.join(source_dir, f)
            new_path = os.path.join(dest_dir, f)
            os.rename(old_path, new_path)


# threshhold()

# png_to_jpeg(
#     '/Users/noahmushkin/codes/cam_finder/data/new_cameras_labels',
#     '/Users/noahmushkin/codes/cam_finder/data/images/new_cameras'
# )

# delete_unused_images(
#     '/Users/noahmushkin/codes/cam_finder/classification/data/new_cameras_labels',
#     '/Users/noahmushkin/codes/cam_finder/classification/data/images/new_cameras'
# )

# round_cam_labels(
#     '/Users/noahmushkin/codes/cam_finder/classification/data/all_round_labels'
#     # '/Users/noahmushkin/codes/cam_finder/classification/data/all_round_labels'
# )

# increase_box_size(
#     '/Users/noahmushkin/codes/cam_finder/classification/data/all_round_labels',
#     '/Users/noahmushkin/codes/cam_finder/classification/data/bigger_round_labels',
#     40
# )

bbox_size_stats(
    '/Users/noahmushkin/codes/cam_finder/classification/data/all_round_labels',
)

# image_filter(
#     '/Users/noahmushkin/Desktop/more_cams',
#     '/Users/noahmushkin/Desktop/processed_pics',
#     '/Users/noahmushkin/Desktop/split_pics'
# )

# delete_unused_xml(
#     '/Users/noahmushkin/codes/cam_finder/classification/data/new_cameras_labels',
#     '/Users/noahmushkin/codes/cam_finder/classification/data/images/new_cameras'
# )

# image_size_check(
#     '/Users/noahmushkin/codes/cam_finder/classification/data/images/new_cameras',
#     '/Users/noahmushkin/codes/cam_finder/classification/data/images/old_cameras'
# )
