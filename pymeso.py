import random
import math
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

author = "Faisal Lawan Muhammad"
version = 0.1
ego= 'frog'


def generate_vertices(j, i, ecell):
    """returns array of a, b, c, d, o(center).
      a vertex of the bottom left corner
      b vertex of the top left corner
      c vertex of the top right corner
      d vertex of the  bottom right corner
      0 vertex of the center
      arguments: j, i, ecell"""

    # vertices
    a = [(i-1)*ecell, (j-1)*ecell]
    b = [i*ecell, (j-1)*ecell]
    c = [i*ecell, j*ecell]
    d = [(i-1)*ecell, j*ecell]

    # center
    o = [(i-0.5)*ecell, (j-0.5)*ecell]

    return [a, b, c, d, o]


def generate_grid(size, num_cell=10, file_name='2D_mesh.txt'):
    """This will generate a grid.
    Mimicking finite element meshing.
    arugments: size of concrete sample(length or breadth), num_cells, file_name
    (where we will store the generate grids and their vertex)"""

    L = size
    H = size
    cell = int(size / num_cell)

    Ldiv = L / cell
    Hdiv = L / cell
    Nelem = Ldiv * Hdiv

    # I will return a object with its key as
    twoD_grid = {}
    twoD = []

    # open a file
    f = open(f"{file_name}", "w")

    for i in range(num_cell):
        i += 1
        for j in range(num_cell):
            j += 1
            # a b c d
            content = twoD_grid[f'{i}, {j}'] = generate_vertices(i, j, cell)
            f.write(f"({i}, {j}): {str(content)} \n")
            twoD.append(generate_vertices(i, j, cell))

    f.close()

    return twoD_grid, twoD


def number_of_polygon_sides(n):
    """probability function for the number of polygon sides n."""

    # both the data below is obtained by fitting on the statistics
    omega = 2.2  # as given in the paper
    nc = 5.8

    return (1/(omega*math.sqrt(math.pi/2)))*math.exp((-2*(n-nc)**2)/omega**2)


def first_angle(n, dkl, dku):
    """After defining an origin inside the aggregate. set the polar
    angle theta of the first corner"""

    # random number between 0 and 1
    nl = random.random()
    angle = (nl/n)*2*math.pi
    # calculate radius
    rad = radius(dkl, dku, nl)

    return angle, rad


def other_angle(n, dkl, dku):
    """
    setting the polar angle and polar radius of the ith
    corner Pi in a similar way.
    """
    nli = random.random()

    denominator = sum([((2/n) + nli) for _ in range(2, n)])
    numerator = ((2*math.pi/n) + (2 * math.pi * nli))
    angle = numerator / denominator

    # calculate radius
    rad = radius(dkl, dku, nli)

    return angle, rad


def radius(dkl, dku, nli):
    return ((dkl+dku)/4) + (2*nli - 1) * ((dku-dkl)/4)


def fuller_curve(d):
    m = 0.5
    # upper limit of aggregate size
    D = 9.5
    return ((d/D)**m)

# This function gets just one pair of coordinates based on the angle theta


def get_circle_coord(theta, x_center, y_center, radius):
    x = radius * math.cos(theta) + x_center
    y = radius * math.sin(theta) + y_center
    return (x, y)


# I want to draw a polygonal aggregate
def draw_polygon():
    """Thus will generate a series of polar angles in theta and radians as well
    as their corresponsing radius. This set of polar coordinates still
    have to be converted to cartesian coordinates  using the polarToCartesian
    function."""

    # aggregate size ranges (lower limit, upper limit)
    aggregate_size_range = [[2.5, 4.75], [
        4.75, 9.5], [9.5, 12.5], [12.5, 20.0]]
    # aggregate_size_range = [[5, 10], [10, 20], [20, 30], [30, 40], [40, 50],
    #  [50, 60], [60, 70], [70, 80]]
    # the number of vertices of a polygon, will be randomly selected
    # aggregate_size_range = [[0.15, 0.300], [0.300, 0.6], [0.6, 1.18],
    #   [1.18, 2.36], [2.36, 4.75], [4.75, 9.5]]

    # this will compute the cumulative probability that an aggregate..
    # passes a sieve opening with a size of d mm.
    probabilities = [fuller_curve(i[1]) for i in aggregate_size_range]
    probabilities = probabilities[::-1]
    print(probabilities)

    # random.choices will randomly select an aggregate size range...
    # while considering that each size range has a weight value...
    # that determines the probability of its occurance.
    size = random.choices(
        [i[1] for i in aggregate_size_range], weights=probabilities)

    num_vertices = [6, 7, 8, 9, 10]
    # select a random vertice number
    n = random.choice(num_vertices)
    # randomly select size range
    size_range = random.choice(aggregate_size_range)
    # lower limit
    di = size_range[0]
    # upper limit
    di1 = size_range[1]

    print(size_range)

    # array of the probability of an n-sided aggregate occuring.
    probabilities = [number_of_polygon_sides(i) for i in num_vertices]
    # randomly picking the n-sides of an aggregates based on the probability function
    n = random.choices(num_vertices, probabilities)[0]
    print(f"sides: {n}")

    # calculate first angle and radius
    fst_angle, first_radius = first_angle(n, di, di1)

    # second angle and radius calculation
    angle_and_radius = []
    print(f'n: {n}, dkl: {di}, dku: {di1}')
    ith = 2
    while ith <= n:
        angle, radius = other_angle(n, di, di1)
        angle_and_radius.append([angle, radius])
        ith += 1

    angle_in_rads = []

    angle_and_radius.insert(0, (fst_angle, first_radius))

    for index, i in enumerate(angle_and_radius):
        if index == 0:
            angle_in_rads.append(i[0])
        else:
            sum_of_angle = sum([i[0] for i in angle_and_radius[:index]])
            angle_in_rads.append(i[0] + sum_of_angle)

    # print(f"Angle in degrees: {angle_in_rads}")
    # print(f"Angle and radius: {angle_and_radius}")

    return angle_and_radius, angle_in_rads


def polarToCartesian(angle_and_radius, angle_in_rads, origin=None):
    """this function converts the generated polar coordinates
    to cartesian coordinates,
    arguments: angle_and_radius, angle_in_rads origin=None"""

    x_coords = []
    y_coords = []
    polygon = []

    for index, i in enumerate(angle_and_radius):
        x_coord = i[1] * np.cos(angle_in_rads[index])
        y_coord = i[1] * np.sin(angle_in_rads[index])

        x_coords.append(x_coord)
        y_coords.append(y_coord)
        # because we have not finished working on x_coords and y_coords
        # polygon.append((x_coord, y_coord))

    if not origin:
        r1, r2 = random.randint(0, 300), random.randint(0, 300)

    origin = (r1, r2)

    # Calculate the coordinates with respect to the origin
    # x_coordinates, y_coordinates = zip(*polygon)
    origin_x, origin_y = origin
    x_coords = [x + origin_x for x in x_coords]
    y_coords = [y + origin_y for y in y_coords]
    for x_coord, y_coord in zip(x_coords, y_coords):
        polygon.append((x_coord, y_coord))

    return x_coords, y_coords, polygon, origin


def plotPolyGon(c_coords):
    """Produces a polygon plot of a set of cartesian coordinates,
    arguments: c_coords (cartesian coordinates)"""

    x_coords, y_coords = c_coords[0], c_coords[1]
    # Create a line plot to connect the points and form a closed polygon
    plt.figure()
    plt.plot(x_coords, y_coords, marker='o')
    plt.plot([x_coords[-1], x_coords[0]], [y_coords[-1], y_coords[0]], 'k-')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Polygon from Set of Angles')
    plt.grid(True)
    plt.axis('equal')
    plt.show()


def plotFilledPolyGonConcrete(c_coords, x_limit=None, y_limit=None):
    """Produces a polygon plot fill of a set of cartesian coordinates,
    arguments: c_coords (cartesian coordinates), x_limit, y_limits of graph"""
    x_coords, y_coords = c_coords[0], c_coords[1]

    # Create a filled polygon
    plt.figure()
    plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Filled Polygon from Set of Angles')
    plt.grid(True)

    if x_limit is not None:
        plt.xlim(x_limit)
    if y_limit is not None:
        plt.ylim(y_limit)

    plt.show()


def hasOverlapWithAxes(x_coords, y_coords):
    """Check for overlap between aggregates and axes of graphs.
    arguments: x_coords, y_coords"""
    # Check if any x or y coordinate is zero or negative
    if any(coord <= 0 for coord in x_coords) or any(coord <= 0 for coord in y_coords) \
            or any(coord >= 300 for coord in x_coords) or any(coord >= 300 for coord in y_coords):
        return True
    else:
        return False


def plotPolyGonConcrete(c_coords, x_limit=None, y_limit=None):
    """Produces a polygon plot fill of a set of cartesian coordinates,
    arguments: c_coords (cartesian coordinates), x_limit, y_limits of graph"""

    x_coords, y_coords = c_coords[0], c_coords[1]

    # Check for overlap with axes
    if hasOverlapWithAxes(x_coords, y_coords):
        print("Polygon vertices overlap with the x or y axis. Adjust coordinates.")
        # return

     # Create a filled polygon
    plt.figure()
    plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Filled Polygon from Set of Angles')
    plt.grid(True)

    if x_limit is not None:
        plt.xlim(x_limit)
    if y_limit is not None:
        plt.ylim(y_limit)

    plt.show()


def doPolygonsOverlap(x_coords1, y_coords1, x_coords2, y_coords2):
    """This functions checks to see if two polygons
    overlap with each other.
    arguments, x, y coordinates of first polygon (aggregate), x,y coordinates of second polygon (aggregate)"""
    poly1 = Polygon(zip(x_coords1, y_coords1))
    poly2 = Polygon(zip(x_coords2, y_coords2))

    return poly1.intersects(poly2)


def generate_aggregates(n, p_coords, c_coords):
    """This function generates a certain number of aggregates.
    arguments(n): number of aggregates, p_coords(polar coordinates), c_coords (cartesian coordinates)"""
    generated_aggregates = []
    for i in range(n):
        p_coords = draw_polygon()
        c_coords = polarToCartesian(p_coords[0], p_coords[1])
        generated_aggregates.append(c_coords)
    return generate_aggregates


def check_for_overlap(gen_aggregates):
    """This will run a loop that gathers the indexes
    of aggregates that overlaps other aggregates so that
    we can remove them. Nox we will have an array of aggregates we can plot
    that do not overlap one another.
    arguments: it takes the array of generated indexes."""
    indexes = []
    for i in range(len(gen_aggregates)):
        first_aggregate = gen_aggregates[i]
        for j in range(i+1, len(gen_aggregates)):
            second_aggregate = gen_aggregates[j]
            # carry out the comparision
            if doPolygonsOverlap(first_aggregate[0], first_aggregate[1], second_aggregate[0], second_aggregate[1]):
                # del non_overlap_generated_aggregates[j]
                indexes.append(j)
                print("Polygon's overlap")
            else:
                print("Polygons do not overlap")

        print(f'Done with round: {i+1}')

    print('end of program')

    non_repeating_indexes = []
    for index in indexes:
        if index not in non_repeating_indexes:
            non_repeating_indexes.append(index)

    non_overlap_aggregates = [i for index, i in enumerate(
        gen_aggregates) if index not in non_repeating_indexes]
    return non_overlap_aggregates


def is_point_inside_polygon(x, y, polygon):
    """this function returns the relationship
    between a set of x, y point and a polygon.
    meaning is  the point  inside or outside the polygon?"""
    n = len(polygon)
    wn = 0  # Initialize winding number

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]  # Wrap around for the last edge

        if y1 <= y:
            if y2 > y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) > 0:
                wn += 1
        elif y2 <= y and (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) < 0:
            wn -= 1

    return wn != 0  # Point is inside if winding number is not zero


def plot_aggregate_only(c_coords, non_overlap_aggregates):
    """this will plot only the aggregates to the graph
    using  the cartesian coords (c_coords) and the non_overlap_aggregates.
    arguments:
    c_coords: c_coords
    non_overlap_aggregates"""

    plt.figure()

    for c_coords in non_overlap_aggregates:

        x_coords, y_coords = c_coords[0], c_coords[1]

        plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Aggregates only plot')
    plt.grid(True)

    if x_limit is not None:
        plt.xlim(x_limit)
    if y_limit is not None:
        plt.ylim(y_limit)

    plt.show()

    return None


def plot_aggregate_with_grids(c_coords, non_overlap_aggregates, cell_size, num_cells,
                              x_limit, y_limit):
    """arguments: cartesian coordinates (c_coords)
    non_overlap_aggregates,
    cell_size,
    num_cells,
    x_limit,
    y_limit"""

    num_cell = num_cells // cell_size

    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, num_cells)
    ax.set_ylim(0, num_cells)

    # Create gridlines for the cells
    for x in range(0, num_cells+1, cell_size):
        ax.axvline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, num_cells+1, cell_size):
        ax.axhline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    # Random colors for each cell
    colors = np.random.rand(num_cell, num_cell, cell_size)
    for i in range(num_cell):
        for j in range(num_cell):
            # cell_color = colors[i, j, :]
            ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j *
                            cell_size], [(j + 1) * cell_size, (j + 1) * cell_size], color='white')

    for c_coords in non_overlap_aggregates:
        x_coords, y_coords = c_coords[0], c_coords[1]
        plt.fill(x_coords, y_coords, facecolor='lightblue', edgecolor='black')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Aggregates with grids plot')

    # Display the plot
    plt.grid(True)
    plt.show()

    return None


def plot_aggregate_with_colored_grids(c_coords, non_overlap_aggregates, cell_size, num_cells,
                                      x_limit, y_limit):
    """arguments: cartesian coordinates (c_coords)
    non_overlap_aggregates,
    cell_size,
    num_cells,
    x_limit,
    y_limit"""

    num_cell = num_cells // cell_size
    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, num_cells)
    ax.set_ylim(0, num_cells)

    # Create gridlines for the cells
    for x in range(0, num_cells+1, cell_size):
        ax.axvline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, num_cells+1, cell_size):
        ax.axhline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    # Random colors for each cell
    colors = np.random.rand(num_cell, num_cell, cell_size)
    for i in range(num_cell):
        for j in range(num_cell):
            # cell_color = colors[i, j, :]
            if i == 0 and j <= 2:
                ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j * cell_size], [
                                (j + 1) * cell_size, (j + 1) * cell_size], color='white')
            else:
                ax.fill_between([i * cell_size, (i + 1) * cell_size], [j * cell_size, j * cell_size], [
                                (j + 1) * cell_size, (j + 1) * cell_size], color='green')

    for c_coords in non_overlap_aggregates:
        x_coords, y_coords = c_coords[0], c_coords[1]
        plt.fill(x_coords, y_coords, facecolor='red', edgecolor='white')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Aggregate with coloured grids')

    # Display the plot
    plt.grid(True)
    plt.show()

    return None


def plot_coloured_grids_only(c_coords, non_overlap_aggregates, cell_size, num_cells,
                             x_limit, y_limit):

    num_cell = num_cells // cell_size

    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, num_cells)
    ax.set_ylim(0, num_cells)

    # Create gridlines for the cells
    for x in range(0, num_cells+1, cell_size):
        ax.axhline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, num_cells+1, cell_size):
        ax.axvline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    # Random colors for each cell
    colors = np.random.rand(num_cell, num_cell, cell_size)
    n = 0
    m = n
    for i in range(num_cell):
        for j in range(num_cell):
            # cell_color = colors[i, j, :]
            if generated_grids[m][5]:
                coloriser = 'red'
            else:
                coloriser = 'white'

            vertices = [
                (((i+1)-1)*cell_size, ((j+1)-1)*cell_size),
                ((i+1)*cell_size, ((j+1)-1)*cell_size),
                ((i+1)*cell_size, (j+1)*cell_size),
                (((i+1)-1)*cell_size, (j+1)*cell_size)
            ]

            # ax.fill_between([(i-1)*ecell, (j-1)*cell_size], [i*cell_size, (j-1)*cell_size], [i*cell_size, j*cell_size], color=coloriser)
            ax.fill_between([v[0] for v in vertices], [v[1]
                            for v in vertices], color=coloriser)
            m += 100
        n += 1
        m = n

    # for c_coords in non_overlap_aggregates:
    #   x_coords, y_coords = c_coords[0], c_coords[1]
    #   plt.fill(x_coords, y_coords, facecolor='white', edgecolor='white')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Material Identification of aggregates only')

    # Display the plot
    plt.grid(True)
    plt.show()

    return None


def plot_aggregate_matrix_itz_grids(c_coords, non_overlap_aggregates, cell_size, num_cells,
                                    x_limit, y_limit):

    num_cell = num_cells // cell_size
    # Create a new figure
    plt.figure()

    # Create axis
    ax = plt.gca()

    # Set the x-axis and y-axis limits
    ax.set_xlim(0, num_cells)
    ax.set_ylim(0, num_cells)

    # Create gridlines for the cells
    for x in range(0, num_cells+1, cell_size):
        ax.axhline(x, color='black', linestyle='-', linewidth=1)

    for y in range(0, num_cells+1, cell_size):
        ax.axvline(y, color='black', linestyle='-', linewidth=1)

    # Fill each cell with a color
    # Random colors for each cell
    colors = np.random.rand(num_cell, num_cell, cell_size)
    n = 0
    m = n
    for i in range(num_cell):
        for j in range(num_cell):
            # cell_color = colors[i, j, :]
            if generated_grids[m][5]:
                coloriser = 'red'
            elif generated_grids[m][7]:
                coloriser = 'green'
            else:
                coloriser = 'white'

            vertices = [
                (((i+1)-1)*cell_size, ((j+1)-1)*cell_size),
                ((i+1)*cell_size, ((j+1)-1)*cell_size),
                ((i+1)*cell_size, (j+1)*cell_size),
                (((i+1)-1)*cell_size, (j+1)*cell_size)
            ]

            # ax.fill_between([(i-1)*ecell, (j-1)*cell_size], [i*cell_size, (j-1)*cell_size], [i*cell_size, j*cell_size], color=coloriser)
            ax.fill_between([v[0] for v in vertices], [v[1]
                            for v in vertices], color=coloriser)
            m += 100
        n += 1
        m = n

    for c_coords in non_overlap_aggregates:
        x_coords, y_coords = c_coords[0], c_coords[1]
        plt.fill(x_coords, y_coords, facecolor='red', edgecolor='white')

    # Add labels and title
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Material Identification of Aggregates, Matrix and ITZ')

    # Display the plot
    plt.grid(True)
    plt.show()

    return None
