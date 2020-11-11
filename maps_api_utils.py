import requests

from settings import MAPS_API_KEY

METADATA_BASE_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata'
PANO_BASE_URL = 'https://www.google.com/maps/@?api=1&map_action=pano'


def pano_exists(lat, lon, heading, pitch=0):
    req_params = {
        'size': '600x300',
        'location': ','.join([str(lat), str(lon)]),
        'fov': 90,
        'heading': heading,
        'pitch': pitch,
        'key': MAPS_API_KEY,
    }
    req = requests.get(url=METADATA_BASE_URL, params=req_params)
    response = req.json()
    return response.get('status', '') == 'OK'


def get_pano_url():
    prev_panos = set()

    def get_viewpoint(lat, lon, heading):
        if not pano_exists(lat, lon, heading):
            return
        hash_string = f'{lat}_{lon}_{heading}'
        if hash_string in prev_panos:
            return
        prev_panos.add(hash_string)
        return f'{PANO_BASE_URL}&viewpoint={lat},{lon}&heading={heading}&pitch=0&fov=90'

    return get_viewpoint
