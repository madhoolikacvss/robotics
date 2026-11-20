def shortest_path(graph, start='S', goal='G'):
    """Find the shortest route using the visibility graph's distance weights."""
    distances = {node: float('inf') for node in graph}
    distances[start] = 0.0
    previous = {}
    unvisited = set(graph)

    while unvisited:
        # Visit the closest remaining node; labels break equal-distance ties.
        node = min(unvisited, key=lambda name: (distances[name], name))
        if distances[node] == float('inf'):
            break
        if node == goal:
            path = [goal]
            while path[-1] != start:
                path.append(previous[path[-1]])
            return list(reversed(path)), distances[goal]

        unvisited.remove(node)
        for neighbor, weight in graph[node].items():
            candidate = distances[node] + weight
            if neighbor in unvisited and candidate < distances[neighbor]:
                distances[neighbor] = candidate
                previous[neighbor] = node
    raise ValueError(f'No path from {start} to {goal}.')
