import matplotlib.pyplot as plt

from osmnx_traverse import OsmnxTraverser
from street_crawler import GraphImageCrawler

# Seattle: [47.597749, 47.620674, -122.358281, -122.322321]
# London: [51.505334, 51.519731, -0.129035, -0.09625]
# Manhattan: [51.505334, 51.519731, -0.129035, -0.09625]
# Upper east side: [40.77611, 40.790277, -73.966121, -73.944307]
# Cambridge Area / MIT: [42.371581, 42.356125, -71.116863, -71.076471]
# Boston Financial District: [42.356207, 42.368895, -71.072855, -71.047126]


# BFS Traversal Visualization Demo:

# loc_list = []


# def print_latlon(lat, lon, bearing, color):
#     loc_list.append((lat, lon, bearing, color))
#     # print(lat, lon, bearing)


# traverser = OsmnxTraverser()
# traverser.load_place_graph(
#     bbox=[42.356207, 42.368895, -71.072855, -71.047126],
#     simple=True
# )
# traverser.bfs_walk(20, print_latlon)
# print(len(loc_list))
# plt.figure()
# plt.scatter(
#     [loc[1] for loc in loc_list],
#     [loc[0] for loc in loc_list],
#     c=[loc[3] for loc in loc_list]
# )

# lengths = [.1 for loc in loc_list]
# plt.quiver(
#     [loc[1] for loc in loc_list],
#     [loc[0] for loc in loc_list],
#     lengths,
#     lengths,
#     angles=[(270 - loc[2]) for loc in loc_list]
# )
# plt.show()

# Image Save Demo:

SAVE_PATH = './data/images/walk_pics/boston_financial'

grapher = GraphImageCrawler(save_path=SAVE_PATH)
grapher.traverse_image_save(
    bbox=[42.356207, 42.368895, -71.072855, -71.047126]
)
