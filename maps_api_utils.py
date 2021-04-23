import requests

from settings import MAPS_API_KEY

METADATA_BASE_URL = 'https://maps.googleapis.com/maps/api/streetview/metadata'
PANO_BASE_URL = 'https://www.google.com/maps/@?api=1&map_action=pano'


def pano_exists(lat, lon, heading, pitch=0, session=None):
    """Uses the google streetview api to check if a pano exists"""

    req_params = {
        'size': '300x300',
        'location': ','.join([str(lat), str(lon)]),
        'fov': 90,
        'heading': heading,
        'pitch': pitch,
        'key': MAPS_API_KEY,
    }
    if session is None:
        req = requests.get(url=METADATA_BASE_URL, params=req_params)
    else:
        req = session.get(url=METADATA_BASE_URL, params=req_params)
    response = req.json()
    return response.get('status', '') == 'OK'


def get_pano_url(lat, lon, heading, session):
    """Returns a street view url for the specified location"""
    try:
        if not pano_exists(lat, lon, heading, session):
            return None
        return f'{PANO_BASE_URL}&viewpoint={lat},{lon}&heading={heading}&pitch=0&fov=90'
    except requests.exceptions.SSLError:
        return None
