from io import BytesIO
from time import sleep, time
import os
from threading import Thread, Lock

from selenium import webdriver
from PIL import Image

from osmnx_traverse import OsmnxTraverser
from maps_api_utils import get_pano_url

DRIVERS = 6


def save_pano(save_path, lat, lon, heading, driver, thread_lock):
    fname = f'{lat}_{lon}_{heading}.jpeg'
    fp = os.path.join(save_path, fname)
    if os.path.isfile(fp):
        return
    url = get_pano_url(lat, lon, heading)
    if url:
        driver.get(url)
        # Wait for page to load fully (url changes on full load)
        while driver.current_url == url:
            sleep(.01)
        with thread_lock:
            screenshot = BytesIO(driver.get_screenshot_as_png())
        screenshot = Image.open(screenshot).convert('RGB')
        screenshot.save(fp=fp)
    else:
        sleep(.1)


class GraphImageCrawler():

    def __init__(self, save_path):
        self.drivers = [webdriver.Chrome() for _ in range(DRIVERS)]
        self.save_path = save_path
        self.shot_lock = Lock()

    def location_save_callback(self, save_list):

        def callback(lat, lon, heading, *args, **kwargs):
            lat = round(lat, ndigits=7)
            lon = round(lon, ndigits=7)
            heading = round(heading, ndigits=None)
            save_list.append((lat, lon, heading))

        return callback

    def traverse_image_save(self, bbox):
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
        start_time = time()
        last_len = len(location_list)
        while location_list:
            try:
                loc = location_list.pop()
                lat, lon, heading = loc[0], loc[1], loc[2]
                save_pano(self.save_path, lat, lon, (heading - 90) % 360, driver, self.shot_lock)
                save_pano(self.save_path, lat, lon, (heading + 90) % 360, driver, self.shot_lock)
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
