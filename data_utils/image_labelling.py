import os
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


# png_to_jpeg(
#     '/Users/noahmushkin/codes/cam_finder/data/new_cameras_labels',
#     '/Users/noahmushkin/codes/cam_finder/data/images/new_cameras'
# )

delete_unused_images(
    '/Users/noahmushkin/codes/cam_finder/classification/data/new_cameras_labels',
    '/Users/noahmushkin/codes/cam_finder/classification/data/images/new_cameras'
)
