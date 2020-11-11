from io import BytesIO
from time import sleep
import os

from selenium import webdriver
from PIL import Image

from osmnx_traverse import OsmnxTraverser
from maps_api_utils import get_pano_url

SAVE_PATH = './walk_pics'


class GraphCamFinder():

    def __init__(self, save_path):
        self.driver = webdriver.Chrome()
        self.save_path = save_path

    def location_save_callback(self, save_list):

        def callback(lat, lon, heading, *args, **kwargs):
            lat = round(lat, ndigits=7)
            lon = round(lon, ndigits=7)
            heading = round(heading, ndigits=None)
            save_list.append((lat, lon, heading))

        return callback

    def save_pano(self):
        get_viewpoint = get_pano_url()

        def save(lat, lon, heading):
            fname = f'{lat}_{lon}_{heading}.jpeg'
            fp = os.path.join(self.save_path, fname)
            if os.path.isfile(fp):
                return
            url = get_viewpoint(lat, lon, heading)
            if url:
                self.driver.get(url)
                sleep(2)
                screenshot = BytesIO(self.driver.get_screenshot_as_png())
                screenshot = Image.open(screenshot).convert('RGB')
                screenshot.save(fp=fp)
            sleep(.2)

        return save

    def traverse_image_save(self, bbox):
        traverser = OsmnxTraverser()
        traverser.load_place_graph(bbox=bbox)
        locations = []
        traverser.bfs_walk(20, self.location_save_callback(locations))
        save_func = self.save_pano()
        for loc_num, loc in enumerate(locations):
            lat, lon, heading = loc[0], loc[1], loc[2]
            save_func(lat, lon, (heading - 90) % 360)
            save_func(lat, lon, (heading + 90) % 360)
            if loc_num % 10 == 0:
                print(round(loc_num / len(locations) * 100))
            print(lat, lon)
