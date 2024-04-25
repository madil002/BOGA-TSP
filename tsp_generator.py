import numpy as np

def generate_tsp(file_path):
    coordinates = []

    with open(file_path, "r") as file:
        in_coord_section = False

        for line in file:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                in_coord_section = True
                continue
            elif line == "EOF":
                break

            if in_coord_section:
                parts = line.split()
                if len(parts) == 3:
                    node, x, y = parts
                    coordinates.append((float(x), float(y)))


    num_cities = len(coordinates)
    distance_matrix = np.zeros((num_cities, num_cities))

    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dx = coordinates[i][0] - coordinates[j][0]
                dy = coordinates[i][1] - coordinates[j][1]
                distance = np.sqrt(dx**2 + dy**2)
                distance_matrix[i][j] = int(round(distance))
                distance_matrix[j][i] = int(round(distance))

    # print(distance_matrix)
    return distance_matrix

generate_tsp("TSPs/wi29.tsp")