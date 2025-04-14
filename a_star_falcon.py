#!/usr/bin/env python3
import numpy as np
import cv2
import heapq
import time
import functools


HEIGHT = 300  # in cm
WIDTH = 600   # in cm

class Node:
    def __init__(self, cost, cost_est, x, y, theta, parent_x, parent_y, parent_theta, rpmL, rpmR):
        self.cost = cost              # Cost so far (in cm)
        self.cost_est = cost_est      # Heuristic cost (Euclidean distance to goal)
        self.x = x                    # x position in cm
        self.y = y                    # y position in cm
        self.theta = theta            # orientation in degrees
        self.parent_x = parent_x      # parent x coordinate
        self.parent_y = parent_y      # parent y coordinate
        self.parent_theta = parent_theta  # parent orientation in degrees
        self.rpmL = rpmL              # left wheel speed (rad/s)
        self.rpmR = rpmR              # right wheel speed (rad/s)

    def __lt__(self, other):
        return self.cost + self.cost_est < other.cost + other.cost_est


def angle_to_index(angle):
    """Convert an angle (in degrees) to an index between 0 and 7 (for 45° resolution)."""
    return int((angle % 360) // 45)

def valid_move(x, y, map_shape, obstacles):
    """Return True if (x,y) is within the bounds of the map and in free space."""
    return 0 <= x < map_shape[1] and 0 <= y < map_shape[0] and obstacles[int(y), int(x)] == 0

def goal_check(x, y, goal_x, goal_y, goal_threshold):
    """Return True if the (x,y) position is within goal_threshold centimeters of the goal."""
    return np.sqrt((goal_x - x)**2 + (goal_y - y)**2) < goal_threshold

def gen_obstacle_map():
    """
    Generates the base obstacle map (BGR image) using simple rectangle shapes.
    The returned image is flipped vertically so that the bottom‐left becomes the origin.
    """
    obstacle_map = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    # Define obstacles (as simple rectangles)
    def l_obstacle1(x, y): return (100 <= x <= 110) and (0 <= y <= 200)
    def l_obstacle2(x, y): return (210 <= x <= 220) and (100 <= y <= 300)
    def l_obstacle3(x, y): return (320 <= x <= 330) and (0 <= y <= 100)
    def l_obstacle4(x, y): return (320 <= x <= 330) and (200 <= y <= 300)
    def l_obstacle5(x, y): return (430 <= x <= 440) and (0 <= y <= 200)
    
    for y in range(HEIGHT):
        for x in range(WIDTH):
            if l_obstacle1(x, y) or l_obstacle2(x, y) or l_obstacle3(x, y) or l_obstacle4(x, y) or l_obstacle5(x, y):
                obstacle_map[y, x] = (0, 0, 255)  # red obstacle
    return np.flipud(obstacle_map)  # flip vertically so bottom-left is the origin

def expand_obstacles(image, radius):
    """
    Expands obstacles by 'radius' pixels using dilation.
    Returns both a display image (BGR) and a grayscale obstacle mask.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 200, 200])
    upper_red = np.array([25, 255, 255])
    obstacle_mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    expanded_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
    
    h, w = expanded_mask.shape
    expanded_mask[:radius+1, :] = 255  # top border
    expanded_mask[h-radius:, :] = 255  # bottom border
    
    output_image = image.copy()
    output_image[np.where(expanded_mask == 255)] = [0, 165, 255]  # orange clearance zone
    output_image[np.where(obstacle_mask == 255)] = [0, 0, 255]      # red obstacles
    return output_image, expanded_mask

def create_new_node(given_Node, revs, goal_x, goal_y, obstacles, delta_time, wheel_radius, wheel_distance):
    """
    Simulates the motion of the robot from 'given_Node' using a pair of wheel velocities.
    Returns a new Node and a flag indicating whether the entire motion (curve) is collision-free.
    
    Parameters:
      - revs: tuple/list (UL, UR) of wheel speeds (in rad/s)
      - wheel_radius and wheel_distance are given in centimeters.
      - delta_time is the simulation time step in seconds.
    """
    def valid_curve(x_arr, y_arr, map_shape, obstacles):
        x_arr = np.round(x_arr).astype(int)
        y_arr = np.round(y_arr).astype(int)
        in_bounds = (0 <= x_arr) & (x_arr < map_shape[1]) & (0 <= y_arr) & (y_arr < map_shape[0])
        x_arr_clipped = np.clip(x_arr, 0, map_shape[1] - 1)
        y_arr_clipped = np.clip(y_arr, 0, map_shape[0] - 1)
        obstacle_free = obstacles[y_arr_clipped, x_arr_clipped] == 0
        return np.all(in_bounds & obstacle_free)
    
    UL, UR = revs
    # Convert wheel dimensions from cm to meters for simulation
    r = wheel_radius / 100.0
    L = wheel_distance / 100.0
    
    Xi = given_Node.x
    Yi = given_Node.y
    Thetai = given_Node.theta  # in degrees

    T = 3  # total simulation time (seconds)
    t = np.arange(0, T + delta_time, delta_time)
    
    # Compute angular velocity and simulate new orientations (using the chosen sign convention)
    w = (r / L) * (UR - UL)
    theta_rad = np.deg2rad(Thetai)
    Thetan = theta_rad - w * t
    v = 0.5 * r * (UL + UR)
    
    dx = v * np.cos(Thetan) * delta_time
    dy = v * np.sin(Thetan) * delta_time
    
    # Integrate to get new positions (convert from m back to cm)
    x_points = np.cumsum(dx) * 100 + Xi
    y_points = np.cumsum(dy) * 100 + Yi
    total_distance = np.sum(np.hypot(np.diff(x_points), np.diff(y_points)))
    
    final_theta = np.rad2deg(Thetan[-1]) % 360
    new_x = x_points[-1]
    new_y = y_points[-1]
    
    cost = given_Node.cost + total_distance
    cost_est = np.sqrt((new_x - goal_x)**2 + (new_y - goal_y)**2)
    
    newNode = Node(cost, cost_est, new_x, new_y, final_theta,
                   given_Node.x, given_Node.y, given_Node.theta, rpmL=UL, rpmR=UR)
    
    map_shape = obstacles.shape
    valid_path = valid_curve(x_points, y_points, map_shape, obstacles)
    return newNode, valid_path

def get_final_path(visited, end_node):
    """
    Backtracks from the end_node using the visited dictionary to produce
    a list of states as (x, y, theta) from start to goal.
    """
    path = []
    current_node = end_node
    path.append((current_node.x, current_node.y, current_node.theta))
    # Loop until the current node is the start (i.e. its parent equals itself)
    while not (current_node.x == current_node.parent_x and 
               current_node.y == current_node.parent_y and 
               current_node.theta == current_node.parent_theta):
        parent_key = (int(round(current_node.parent_y)), 
                      int(round(current_node.parent_x)), 
                      angle_to_index(current_node.parent_theta))
        if parent_key not in visited:
            break
        current_node = visited[parent_key]
        path.append((current_node.x, current_node.y, current_node.theta))
    path.reverse()
    return path

def compute_relative_moves(path_states):
    """
    Given a list of states (x, y, theta) along the planned path, compute the
    relative movements (dx, dy, dtheta) between consecutive states.
    dtheta is converted to radians.
    """
    moves = []
    for i in range(len(path_states) - 1):
        x1, y1, theta1 = path_states[i]
        x2, y2, theta2 = path_states[i+1]
        dx = x2 - x1
        dy = y2 - y1
        dtheta = np.deg2rad(theta2 - theta1)
        moves.append([dx, dy, dtheta])
    return moves

def A_star_search(obstacles, start, end, Revs, delta_time, wheel_radius, wheel_distance, goal_threshold):
    """
    Modified A* search that uses the given obstacle mask.
    Input 'start' should be (x, y, theta) and 'end' should be (x, y), both in cm
    (with a bottom-left origin). Note that the A* algorithm internally converts the y-values
    to a coordinate system consistent with our image (top-left origin).
    """
    height, width = obstacles.shape
    start_x, start_y, start_theta = start
    end_x, end_y = end

    # Convert to image coordinate system (flip y)
    start_y_img = height - start_y
    end_y_img = height - end_y

    start_node = Node(0, 0, start_x, start_y_img, start_theta,
                      start_x, start_y_img, start_theta, 0, 0)

    open_set = []
    heapq.heappush(open_set, start_node)
    
    seen = np.full((height, width, 8), False, dtype=bool)
    seen[int(start_y_img), int(start_x), angle_to_index(start_theta)] = True
    visited = {}
    visited[(int(start_y_img), int(start_x), angle_to_index(start_theta))] = start_node
    closed_set = np.full((height, width, 8), False, dtype=bool)
    
    RPM1, RPM2 = Revs
    directions = [[0, RPM1], [RPM1, 0], [RPM1, RPM1],
                  [0, RPM2], [RPM2, 0], [RPM2, RPM2],
                  [RPM1, RPM2], [RPM2, RPM1]]
    
    while open_set:
        current_node = heapq.heappop(open_set)
        cx, cy, ctheta = current_node.x, current_node.y, current_node.theta
        idx = (int(round(cy)), int(round(cx)), angle_to_index(ctheta))
        if closed_set[idx]:
            continue
        closed_set[idx] = True
        
        # Check if the goal has been reached (adjusting the goal's y from bottom-up)
        if goal_check(cx, cy, end_x, height - end_y, goal_threshold):
            return get_final_path(visited, current_node)
        
        # Explore all available motions
        for rev in directions:
            new_node, valid = create_new_node(current_node, rev, end_x, height - end_y,
                                              obstacles, delta_time, wheel_radius, wheel_distance)
            if valid:
                new_idx = (int(round(new_node.y)), int(round(new_node.x)), angle_to_index(new_node.theta))
                if not closed_set[new_idx]:
                    if not seen[new_idx]:
                        seen[new_idx] = True
                        visited[new_idx] = new_node
                        heapq.heappush(open_set, new_node)
                    else:
                        if visited[new_idx].cost > new_node.cost:
                            visited[new_idx] = new_node
                            heapq.heappush(open_set, new_node)
    return None  # No valid path was found


def plan_path(start, end, robot_radius, clearance, delta_time, goal_threshold,
              wheel_radius, wheel_distance, rpm1, rpm2):
    """
    Plans a path from start to end using an A* search algorithm.
    
    Parameters:
      start: tuple (x, y, theta) in cm and degrees (with the origin at bottom-left)
      end: tuple (x, y) in cm (bottom-left)
      robot_radius: robot's radius in cm
      clearance: clearance (in cm) to be maintained from obstacles
      delta_time: simulation time step (seconds) for the motion model
      goal_threshold: distance threshold (in cm) for goal attainment
      wheel_radius: radius of a wheel in cm
      wheel_distance: distance between the two wheels in cm
      rpm1, rpm2: two wheel speed values (in rad/s) to form the action set
      
    Returns:
      A list of motion commands in the format [[dx, dy, dtheta], ...]
      where dx and dy are in centimeters and dtheta is in radians.
      Returns None if no path is found.
    """
    # Generate the base obstacle map
    obs_map = gen_obstacle_map()
    # Inflate obstacles to account for robot dimensions and clearance.
    # (Here we use the provided radii as the dilation radius in pixels.)
    _, obs_mask = expand_obstacles(obs_map, int(robot_radius))
    _, obs_mask = expand_obstacles(obs_map, int(clearance))
    
    # Run the A* search on the obstacle mask
    path_states = A_star_search(obs_mask, start, end, (rpm1, rpm2),
                                delta_time, wheel_radius, wheel_distance, goal_threshold)
    if path_states is None:
        print("No path found!")
        return None
    
    # Convert the sequence of states to relative motions.
    actions = compute_relative_moves(path_states)
    return actions


if __name__ == "__main__":
    # Define start (x, y, theta in degrees) and end (x, y) positions in cm.
    start = (50, 50, 0)    # Starting at (50, 50) with 0° orientation (bottom-left origin)
    end = (550, 250)       # Goal location
    
    # Set parameters (tweak as appropriate)
    robot_radius = 22      # in cm
    clearance = 5          # in cm
    delta_time = 0.1       # seconds (simulation time step)
    goal_threshold = 5     # in cm; how close you need to get to the goal
    wheel_radius = 3.3     # in cm (e.g., 33 mm)
    wheel_distance = 28.7  # in cm (e.g., 287 mm)
    rpm1 = 5               # wheel speed option 1 (rad/s)
    rpm2 = 7               # wheel speed option 2 (rad/s)
    
    # Call the planner and print the list of relative moves
    path_commands = plan_path(start, end, robot_radius, clearance, delta_time,
                              goal_threshold, wheel_radius, wheel_distance, rpm1, rpm2)
    if path_commands is not None:
        print("Planned path (relative moves [dx, dy, dtheta in radians]):")
        for move in path_commands:
            print(move)
