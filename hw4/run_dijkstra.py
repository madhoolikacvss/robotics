"""Calculate the shortest path from the current map geometry."""

from hw4.visibility_graph import build_visibility_graph
from hw4.dijkstra import shortest_path
from hw4.path_waypoints import path_to_waypoints


if __name__ == '__main__':
    nodes, graph = build_visibility_graph()
    path, distance = shortest_path(graph)
    print('Shortest path:', ' -> '.join(path))
    print(f'Distance: {distance:.3f} m')
    print('Waypoints (x meters, y meters, yaw radians):')
    for label, (x, y, yaw) in zip(path, path_to_waypoints(nodes, path)):
        print(f'  {label}: ({x:.6f}, {y:.6f}, {yaw:.6f})')
