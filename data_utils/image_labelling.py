import os
import json
import xml.etree.ElementTree as etree

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


# threshhold()

# png_to_jpeg(
#     '/Users/noahmushkin/codes/cam_finder/data/new_cameras_labels',
#     '/Users/noahmushkin/codes/cam_finder/data/images/new_cameras'
# )

delete_unused_images(
    '/Users/noahmushkin/codes/cam_finder/classification/data/new_cameras_labels',
    '/Users/noahmushkin/codes/cam_finder/classification/data/images/new_cameras'
)

# delete_unused_xml(
#     '/Users/noahmushkin/codes/cam_finder/classification/data/new_cameras_labels',
#     '/Users/noahmushkin/codes/cam_finder/classification/data/images/new_cameras'
# )

# image_size_check(
#     '/Users/noahmushkin/codes/cam_finder/classification/data/images/new_cameras',
#     '/Users/noahmushkin/codes/cam_finder/classification/data/images/old_cameras'
# )
