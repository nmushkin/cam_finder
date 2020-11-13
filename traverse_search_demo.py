import matplotlib.pyplot as plt

from osmnx_traverse import OsmnxTraverser
from street_crawler import GraphImageCrawler


# BFS Traversal Visualization Demo:

# loc_list = []


# def print_latlon(lat, lon, bearing, color):
#     loc_list.append((lat, lon, bearing, color))
#     # print(lat, lon, bearing)


# traverser = OsmnxTraverser()
# traverser.load_place_graph(
#     bbox=[47.597749, 47.620674, -122.358281, -122.322321],
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

SAVE_PATH = './data/walk_pics'

grapher = GraphImageCrawler(save_path=SAVE_PATH)
grapher.traverse_image_save(
    bbox=[47.597749, 47.620674, -122.358281, -122.322321]
)