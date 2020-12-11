from io import BytesIO, StringIO
from time import sleep, time
import os
from threading import Thread

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
from requests import Session

from osmnx_traverse import OsmnxTraverser
from maps_api_utils import get_pano_url

DRIVERS = 4


def save_pano(save_path, lat, lon, heading, driver, session):
    """Loads and saves a streetview image for a given location"""

    fname = f'{lat}_{lon}_{heading}.jpeg'
    fp = os.path.join(save_path, fname)
    if os.path.isfile(fp):
        return
    url = get_pano_url(lat, lon, heading, session)
    if url:
        driver.get(url)
        # Wait for page to load fully (url updates on full load)
        while driver.current_url == url:
           sleep(.01)
        sleep(.5)
        screenshot = driver.get_screenshot_as_png()
        screenshot = Image.open(BytesIO(screenshot)).convert('RGB')
        screenshot.save(fp=fp)
    else:
        sleep(.1)


class GraphImageCrawler():

    def __init__(self, save_path):
        options = Options()
        # options.add_argument('--headless')
        # options.add_argument('--window-size=1600x1600')
        self.drivers = [
            webdriver.Chrome(chrome_options=options) for _ in range(DRIVERS)
        ]
        self.save_path = save_path

    def location_save_callback(self, save_list):

        def callback(lat, lon, heading, *args, **kwargs):
            lat = round(lat, ndigits=7)
            lon = round(lon, ndigits=7)
            heading = round(heading, ndigits=None)
            save_list.append((lat, lon, heading))

        return callback

    def traverse_image_save(self, bbox):
        """Generates a list of locations and gets imagery of them"""

        traverser = OsmnxTraverser()
        traverser.load_place_graph(bbox=bbox)
        locations = []
        traverser.bfs_walk(30, self.location_save_callback(locations))
        print(f'Total Points: {len(locations)}')
        threads = []
        for thread_num in range(DRIVERS):
            threads.append(
                Thread(target=self.thread_save,
                       args=(locations,
                             self.drivers[thread_num],
                             thread_num == 0))
            )
        for t in threads:
            t.start()
            sleep(.5)
        for t in threads:
            t.join()

    def thread_save(self, location_list, driver, log=False):
        """Per-thread function to save streetview imagery"""

        session = Session()
        start_time = time()
        last_len = len(location_list)
        while location_list:
            try:
                loc = location_list.pop()
                lat, lon, heading = loc[0], loc[1], loc[2]
                save_pano(self.save_path, lat, lon, (heading - 90) % 360, driver, session)
                save_pano(self.save_path, lat, lon, (heading + 90) % 360, driver, session)
                print(lat, lon)
            except IndexError:
                return

            if log:
                time_diff = time() - start_time
                if time_diff >= 10:
                    diff = last_len - len(location_list)
                    print(f'{round((time_diff) / (diff + 1))} s')
                    start_time = time()
                    last_len = len(location_list)
