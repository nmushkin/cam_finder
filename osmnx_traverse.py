import os
from collections import deque
from math import floor, cos, sin, atan2, pi, log, tan

import osmnx as ox
import matplotlib.pyplot as plt


GRAPH_DIR = '.data/osmnx_graphs/'


class OsmnxTraverser():

    def __init__(self):
        self.G = None

    def load_place_graph(self, place_name=None, bbox=None, simple=True):
        """Loads graphml file from storage if exists or downloads one"""

        if not os.path.isdir(GRAPH_DIR):
            os.makedirs(GRAPH_DIR)
        filename = f'{place_name or bbox}_{str(simple)}.graphml'
        g_filename = os.path.join(GRAPH_DIR, filename)
        if os.path.isfile(g_filename):
            G = ox.io.load_graphml(g_filename)
        else:
            if place_name:
                G = ox.graph_from_place(place_name,
                                        simplify=simple,
                                        network_type='drive')
            elif bbox:
                G = ox.graph_from_bbox(*bbox,
                                       simplify=simple,
                                       network_type='drive')
            ox.io.save_graphml(G, g_filename)
        self.G = G

    def bfs_walk(self, distance, node_callback):
        """Performs a bfs walk on the graph, calling callback along the way"""

        discovered = deque()
        print(len(self.G.nodes))
        first_node = next(iter(self.G.nodes))
        self.G.nodes[first_node]['level'] = 0
        discovered.appendleft(first_node)
        visited = set([first_node])
        taken_edges = {e: set() for e in iter(self.G.nodes)}

        while(discovered):
            if len(visited) % 20 == 0:
                print(f'{round(len(visited) / len(self.G.nodes) * 100)}%')
            node_num = discovered.pop()
            bfs_visit(self.G,
                      node_num,
                      visited,
                      discovered,
                      node_callback,
                      taken_edges,
                      distance)

    def dfs_walk(self, distance, node_callback):
        """Performs a dfs walk on the graph, calling callback along the way"""

        first_node = next(iter(self.G.nodes))
        self.G.nodes[first_node]['level'] = 0
        to_visit = [first_node]
        visited = set([first_node])
        while to_visit:
            node_num = to_visit.pop()
            if node_num not in visited:
                visited.add(node_num)

            node_info = self.G.nodes[node_num]
            my_level = node_info['level']
            neighbors = self.G.adj[node_num]
            for n in neighbors:
                if n not in visited:
                    edge_info = neighbors[n]
                    distance_to = floor(edge_info[0]['length'])
                    n_info = self.G.nodes[n]
                    n_level = n_info.get('level', None)
                    if n_level is None or n_level > my_level + 1:
                        n_info['level'] = my_level + 1
                    n_stops = floor(distance_to / distance)
                    stops = stops_between(self.G, n_stops, node_num, n)
                    for stop in stops:
                        # Callback with lat, lon, bearing and level
                        node_callback(stop[0], stop[1], stop[2],
                                      n_info['level'])
                    to_visit.append(n)


def bfs_visit(G, node_num, visited, discovered, callback, paths, distance):
    node_info = G.nodes[node_num]
    my_level = node_info['level']
    neighbors = G.adj[node_num]
    for n in neighbors:
        n_info = G.nodes[n]
        n_level = n_info.get('level', None)
        if n_level is None or n_level > my_level + 1:
            n_info['level'] = my_level + 1

        if n not in paths[node_num]:
            edge_info = neighbors[n][0]
            if node_num in paths[n]:
                path_stops = stops(edge_info, node_info, n_info, distance,
                                   True)
            else:
                path_stops = stops(edge_info, node_info, n_info, distance)

            for stop in path_stops:
                # Callback with lat, lon, bearing and level
                callback(stop[0], stop[1], stop[2], n_info['level'])

            paths[node_num].add(n)

        if n not in visited:
            discovered.appendleft(n)
            visited.add(n)


def stops(edge_info, from_node, to_node, min_distance, first_only=False):
    """Returns a set of points bewteen from and to nodes

       The returned nodes will be approximately min_distance meters apart
    """
    if edge_info.get('geometry'):
        point_stops = string_stops(
            edge_info['geometry'],
            edge_info['length'],
            min_distance,
            first_only)
    else:
        lat_1, lon_1 = from_node['y'], from_node['x']
        lat_2, lon_2 = to_node['y'], to_node['x']
        distance_to = edge_info['length']
        n_stops = 0 if first_only else floor(distance_to / min_distance)
        point_stops = stops_between(n_stops, lat_1, lon_1, lat_2, lon_2)

    return point_stops


def stops_between(n_stops, lat_1, lon_1, lat_2, lon_2, include_start=True):
    """Returns n_stops points linearly between two points"""

    bearing = bearing_between(lat_1, lon_1, lat_2, lon_2)
    lat_dist = (lat_2 - lat_1) / (n_stops + 1)
    lon_dist = (lon_2 - lon_1) / (n_stops + 1)
    stops = []
    start = 0 if include_start else 1
    for stop in range(start, n_stops + 1):
        lat = lat_1 + (lat_dist * stop)
        lon = lon_1 + (lon_dist * stop)
        stops.append((lat, lon, bearing))
    return stops


def line_string_great_circle(line_string):
    """Calculate the great-circle distance of a LineString of (lon, lat)"""
    coords = list(line_string.coords)
    distance = 0
    for coord in range(1, len(coords)):
        lon_1, lat_1 = coords[coord - 1][0], coords[coord - 1][1]
        lon_2, lat_2 = coords[coord][0], coords[coord][1]
        distance += ox.distance.great_circle_vec(lat_1, lon_1, lat_2, lon_2)
    return distance


def string_stops(line_string, true_length, min_distance, first_only=False):
    """Returns points from a shapely line string at least min_distance apart"""
    stops = []
    line_length = line_string_great_circle(line_string)
    coords = list(line_string.coords)
    start_p = 0
    point_num = 1
    for point in coords[1:]:
        lon_1, lat_1 = coords[start_p][0], coords[start_p][1]
        lon_2, lat_2 = point[0], point[1]
        curr_dist = ox.distance.great_circle_vec(lat_1, lon_1, lat_2, lon_2)
        true_dist = (curr_dist / line_length) * true_length
        if first_only:
            return stops_between(0, lat_1, lon_1, lat_2, lon_2)
        elif true_dist >= min_distance:
            num_stops = floor(true_dist / min_distance)
            new_stops = stops_between(num_stops, lat_1, lon_1, lat_2, lon_2)
            coords[point_num] = (new_stops[-1][1], new_stops[-1][0])
            if start_p > 0:
                stops.pop()
            stops.extend(new_stops)
            start_p = point_num
        point_num += 1
    if not stops:
        stops = stops_between(0, lat_1, lon_1, lat_2, lon_2)
    return stops


def bearing_between(lat1, lon1, lat2, lon2):
    return ox.bearing.get_bearing((lat1, lon1), (lat2, lon2))
