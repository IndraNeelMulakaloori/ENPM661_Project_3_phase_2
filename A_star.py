import numpy as np
import cv2 as cv
import heapq
import copy
import time
import functools
import sys
import math
from queue import PriorityQueue
sys.setrecursionlimit(2000)

## Constants do not modify
SCALE_FACTOR = 2
HEIGHT = 250*SCALE_FACTOR
WIDTH = 600*SCALE_FACTOR
RED = (0, 0, 255)  # Outline color  
ORANGE = (0, 165, 255) # Fill color in BGR
GRAY = (128,128,128) # Background color
MAGENTA = (255,0,255) # Initial Position
GREEN = (255,255,0) # FINAL POSITION
PINK = (170,51,106) # BACKTRACKING COLOR
BROWN = [88,57,39] # NODE Traversal COLOR


## turtle 3 wafflepi 
# Wheel Radius (R): 33 mm
# Robot Radius (r): 220 mm
# Wheel Distance (L): 287 mm

# Timer decorator to measure execution time of functions
def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # Start time
        result = func(*args, **kwargs)  # Execute the wrapped function
        end_time = time.perf_counter()  # End time
        run_time = end_time - start_time  # Calculate runtime
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return result  # Return the result of the wrapped function
    return wrapper



# gen_obstacle_map generates the map and its obstacles using half planes 
# and semi-algebraic models. Each obstacle is composed of a union of convex
# polygons that define it. It then constructs an in image in BGR and sets
# obstacle pixels as red in the image. Additionally, the entire obstacle map
# can be configured for a certain resolution by the given scale factor, SCALE_FACTOR.
# When SCALE_FACTOR = 1, each pixel represents 1 mm. SCALE_FACTOR = 10, each pixel represents .1 mm.
def gen_obstacle_map(sf=SCALE_FACTOR):
    # Set the height and width of the image in pixels.
    height = HEIGHT
    width = WIDTH
    # Create blank canvas.
    obstacle_map = np.zeros((height,width,3), dtype=np.uint8 )

    # Arbitrary increase in size of obstacles to fit new expanded map size. Map size was height = 50 and width = 180
    # in prior project. This makes the map more filled with obstacles by expanding their size. 
    sf=sf*1
    
    # Define polygons for line obstacles.
    def l_obstacle1(x,y):
        return (100*sf <= x <= 110*sf) and (0*sf <= y <= 200*sf)
    
    def l_obstacle2(x,y):
        return (210*sf <= x <= 220*sf) and (100*sf <= y <= 300*sf)
    
    def l_obstacle3(x,y):
        return (320*sf <= x <= 330*sf) and (0*sf <= y <= 100*sf)
    
    def l_obstacle4(x,y):
        return (320*sf <= x <= 330*sf) and (200*sf <= y <= 300*sf)
    
    def l_obstacle5(x,y):
        return (430*sf <= x <= 440*sf) and (0*sf <= y <= 200*sf)
  
    # For every pixel in the image, check if it is within the bounds of any obstacle.
    # If it is, set it's color to red.
    for y in range(height):
        for x in range(width):
            if (l_obstacle1(x, y) or l_obstacle2(x,y) or l_obstacle3(x,y) or l_obstacle4(x,y) 
                or l_obstacle5(x,y)):
                obstacle_map[y, x] = (0, 0, 255) 
            

    # The math used assumed the origin was in the bottom left.
    # The image must be vertically flipped to satisy cv2 convention. 
    return np.flipud(obstacle_map)

# expand_obstacles takes the obstacle map given by gen_obstacle_map as an image, along with
# the scale factor SCALE_FACTOR, and generates two images. The first output_image, is a BGR image
# to draw on used for visual display only. expanded_mask is a grayscale image with white
# pixels as either obstacles or clearance space around obstacles. This function will take 
# the given obstacle image and apply a specified radius circular kernel to the image. This ensures
# an accurate clearance around every obstacle.
def expand_obstacles(image, scale_factor, radius):

    radius = scale_factor*radius

    # Convert image to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Define color mask for red and create grayscale image.
    lower_red = np.array([0, 200, 200])
    upper_red = np.array([25, 255, 255])
    obstacle_mask = cv.inRange(hsv, lower_red, upper_red)
    
    # Create circular structuring element for expansion
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    # Apply kernel to get 2 mm dilation around all elements.
    expanded_mask = cv.dilate(obstacle_mask, kernel, iterations=1)

    # Apply 2 mm dilation to all of the borders.
    h, w = expanded_mask.shape
    expanded_mask[:radius+1, :] = 255  # Top border
    expanded_mask[h-radius:, :] = 255  # Bottom border
    expanded_mask[:, :radius+1] = 255  # Left border
    expanded_mask[:, w-radius:] = 255  # Right border
    
    # Create the output image and apply color orange to all obstacle and clearance
    # pixels.
    output_image = image.copy()
    output_image[np.where(expanded_mask == 255)] = [0, 165, 255]  # Color orange
    
    # Restore original red pixels. This creates an image with red obstacles,
    # and orange clearance zones. 
    output_image[np.where(obstacle_mask == 255)] = [0, 0, 255]  
    
    return output_image, expanded_mask

# prompt the user for a point. prompt is text that specifies what
# type of point is be given. prompt is solely used for terminal text output.
# SCALE_FACTOR is the scale factor to ensure the user's input is scaled correctly for the map. 
# image is passed to ensure the point is within the image bounds. obstacles is passed
# to ensure the user's point does not lie in an obstacle. The function returns the user's
# points as integers. It also prompts the user for an angle and ensures that the angle
# is a multiple of 30. 
def get_point(prompt, SCALE_FACTOR, bloated_map):
    valid_input = False
    
    while not valid_input:
        # Get x and y input and adjust by scale factor SCALE_FACTOR.
        try:
            x = int(input(f"Enter the x-coordinate for {prompt} (int): "))
            y = int(input(f"Enter the y-coordinate for {prompt} (int): "))
        except ValueError:
            print("Invalid input. Please enter a numerical value.")
            continue
        
        # Ensure theta meets constraints.
        if prompt == "start":
            while True:
                try:
                    theta = int(input(f"Enter the theta-coordinate for {prompt} (must be 0-360 and a multiple of 30): "))
                    if 0 <= theta <= 360 and theta % 30 == 0:
                        break
                    else:
                        print("Invalid theta. It must be between 0 and 360 and a multiple of 30. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter an integer value for theta.")
        
        # Correct the y value to account for OpenCV having origin in top left.
        obstacle_y = HEIGHT - y*SCALE_FACTOR
        
        # Validate position against obstacles
        if boundary_check(x*SCALE_FACTOR, obstacle_y, bloated_map):
            valid_input = True
        else:
            print("Invalid Input. Within Obstacle. Please try again.")
    if prompt == "start":
        return int(x*SCALE_FACTOR), int(obstacle_y), int(theta)
    else:
        return int(x*SCALE_FACTOR), int(obstacle_y)


def heuristics(current_node : tuple, final_node : tuple):
        point_A = np.array([current_node[0],current_node[1]])
        point_B = np.array([final_node[0],final_node[1]])
        return np.linalg.norm(point_A - point_B)

def discretize_nodes(node : tuple, threshold_xy = 0.5 , threshold_theta = 30 ) -> tuple:
    x_index = int(node[0] / threshold_xy)
    y_index = int(node[1] / threshold_xy)

    theta = node[2] % 360
    if abs(theta - 360) < 1e-3 or abs(theta) < 1e-3:
        theta = 0  # Snap near-360 or near-0 to 0
    theta_index = int(theta / threshold_theta)

    # Ensure angle index is capped at max
    max_theta_idx = int(360 / threshold_theta) - 1
    if theta_index > max_theta_idx:
        theta_index = max_theta_idx

    return (x_index, y_index, theta_index)

def boundary_check(row_index  : int, col_index : int, bloated_map : list) -> bool:
    """
    Checks if a given position is within the boundaries of the map and not blocked.

    Args:
        row_index (int): The row index (y-coordinate) of the position.
        col_index (int): The column index (x-coordinate) of the position.
        bloated_map (list): The map containing obstacles and bloated regions.

    Returns:
        bool: True if the position is valid (within bounds and not blocked), False otherwise.
    """
    # Check if the position is out of bounds
    if row_index < 0 or row_index >= HEIGHT or col_index < 0 or col_index >= WIDTH:
        # Uncomment the line below for debugging out-of-bounds positions
        # print(f"Out of bounds: {row_index},{col_index} (Invalid Bounds)")
        return False

    # # Check if the position is blocked (part of an obstacle or bloated region)
    if np.all(bloated_map[int(row_index), int(col_index)] == RED) or np.all(bloated_map[int(row_index), int(col_index)] == ORANGE):
        # Uncomment the line below for debugging blocked positions
        print(f"Blocked or visited: {row_index},{col_index}")
        return False

    # If the position is valid, return True
    return True
def get_valid_rpms():
    while True:
        try:
            user_input = input("Enter the left and right wheel RPMs (space-separated, e.g., 5 10): ")
            rpm_parts = user_input.strip().split()

            if len(rpm_parts) != 2:
                print(" Please enter exactly two values.")
                continue

            rpm1 = int(rpm_parts[0])
            rpm2 = int(rpm_parts[1])
            if rpm1 < 0 or rpm2 < 0:
                print("RPM values must be non-negative.")
                continue
            return rpm1, rpm2

        except ValueError:
            print("Invalid input. Please enter integers only.")

def cost(Xi,Yi,Thetai,UL,UR,bloated_map):
   t = 0
   r = 0.038
   L = 0.354
   dt = 0.1
   Xn=Xi
   Yn=Yi
   Thetan = 3.14 * Thetai / 180
   D=0
   # OpenCV uses (col, row) or (x, y)
   while t<1:
        t = t + dt
        Delta_Xn = 0.5*r * (UL + UR) * math.cos(Thetan) * dt
        Delta_Yn = 0.5*r * (UL + UR) * math.sin(Thetan) * dt
        Thetan += (r / L) * (UR - UL) * dt
        Xn += Delta_Xn   
        Yn += Delta_Yn   
        D  +=math.hypot(Delta_Xn,Delta_Yn)
        
        
   
   Thetan = math.degrees(Thetan) % 360

        # Convert to OpenCV coordinates

   if boundary_check(Xn ,Yn ,bloated_map):
    return (Xn  , Yn , Thetan, D)

   else:
       return None

@timer
def A_star(initial_position : tuple, final_position : tuple, wheel_rpms : tuple, bloated_map : list[list],video_output : object):
    
    new_node_info = []
    initial_c2g =int(heuristics(initial_position,final_position))
    
    threshold_xy,threshold_theta = 0.7,30
    
    discretized_initial_position = discretize_nodes(initial_position)
    new_node_info.append({
        'node' : initial_position,
        'parent_node_index' : -1 ,
        'c2c' : 0,
        'c2g' : initial_c2g,
        'total_cost' : initial_c2g
        
    })
    open_set = PriorityQueue()
    open_set.put((new_node_info[0]['total_cost'],initial_position,0))
    
    # print(open_set.queue)
     
    
    visited_matrix = np.ones((int(HEIGHT/threshold_xy),int(WIDTH/threshold_xy),int(360/threshold_theta)),dtype='int') * -1
    visited_matrix[discretized_initial_position[0]][discretized_initial_position[1]][discretized_initial_position[2]] = 0
    
    closed_list = set()
    goal_threshold = 1.5
    
    video_frame_counter = 0  # Counter to control video frame writing
    while not open_set.empty():
        
        _ , parent_node,parent_index = open_set.get()
        if ((final_position[0] - parent_node[0]) ** 2 + (final_position[1] - parent_node[1]) ** 2) <= goal_threshold ** 2:
            print("Final Node Found")
            return parent_index,new_node_info
            
        # parent_node = discretize_nodes(parent_node)
        if parent_node in closed_list:
            continue
        
        closed_list.add(parent_node)
        # Write frames to the video at intervals
        
        if video_frame_counter % 300 == 0:
            cv.circle(bloated_map, (initial_position[1], initial_position[0]), 3, MAGENTA, -1)
            cv.circle(bloated_map, (final_position[1], final_position[0]), 3, GREEN, -1)
            video_output.write(bloated_map)
        video_frame_counter += 1
        
        
        
        # video_output.write(bloated_map)
        moves = [[0,wheel_rpms[0]],[wheel_rpms[0],0],[wheel_rpms[0],wheel_rpms[0]],
                 [0,wheel_rpms[1]],[wheel_rpms[1],0],[wheel_rpms[1],wheel_rpms[1]],
                 [wheel_rpms[0],wheel_rpms[1]],[wheel_rpms[1],wheel_rpms[0]]]
        for move in moves:
            ### cost() is much more quicker than cost_numpy()
            new_node = cost(parent_node[0],parent_node[1],parent_node[2],move[0],move[1],bloated_map)
            if new_node is not None:
                new_X,new_Y,new_theta,cost_to_come = new_node
                cv.arrowedLine(bloated_map,(int(parent_node[1]),int(parent_node[0])),(int(new_node[1]),int(new_node[0]) ),color=(255,255,255),thickness=1)
                # print((new_X,new_Y,new_theta))
                discretized_new_node = discretize_nodes((new_X,new_Y,new_theta),threshold_xy=threshold_xy,threshold_theta=threshold_theta)
                if new_node not in closed_list:
                    # print(discretized_new_node)
                    visited_index = visited_matrix[discretized_new_node[0]][discretized_new_node[1]][discretized_new_node[2]]
                    # print(visited_index)
                    if visited_index == -1 :
                        cost_to_go = heuristics(new_node,final_position)
                        total_cost = new_node_info[parent_index]['c2c'] + cost_to_come + cost_to_go
                        new_node_info.append({
                                'node' : new_node,
                                'parent_node_index' : parent_index,
                                'c2c' : new_node_info[parent_index]['c2c'] + cost_to_come,
                                'c2g' :  cost_to_go,
                                'total_cost' :  total_cost
                            }
                        )

                        visited_matrix[discretized_new_node[0]][discretized_new_node[1]][discretized_new_node[2]] = len(new_node_info) - 1
                        open_set.put((total_cost,new_node,len(new_node_info) - 1))
                        

                    elif new_node_info[visited_index]['c2c'] > cost_to_come + new_node_info[parent_index]['c2c']:
                        new_node_info[visited_index]['c2c']   =  cost_to_come + new_node_info[parent_index]['c2c']
                        new_node_info[visited_index]['total'] = new_node_info[visited_index]['c2c'] + new_node_info[visited_index]['c2g']
                        new_node_info[visited_index]['parent_node_index'] = parent_index
                        # new_node_info[visited_index]['discretized_parent_node'] = discretize_parent_node
                        open_set.put((new_node_info[visited_index]['total_cost'],new_node,visited_index))
                    
                    
    # print(len(new_node_info))
    return -1
def backtrack(node_index : int, node_info_list : list, backtrack_list: list) -> list:
    """
    Function to backtrack the path from the final node to the
    intial node

    Args:
        node_index (int): Current node index
        node_info_list (list): List of node information
        backtrack_list (list): List to store the backtracking path

    Returns:
        list: returns the backtracking path list
    """
    if node_index == -1:
        ## If the node index is 0 then return the backtrack list
        # as the intial node is reached
        return backtrack_list
    
    ## Recursively call the backtrack function to get the path
    backtrack_list = backtrack(node_info_list[node_index]['parent_node_index'],node_info_list,backtrack_list)
    ## Append the current node state to the backtrack list
    backtrack_list.append(node_info_list[node_index]['node'])
    ## Return the backtrack list
    return backtrack_list 
def create_video():
    """
    Creates and initializes a VideoWriter object for saving the traversal video.

    This function sets up the video codec, filename, frame rate, and resolution for the output video.

    Returns:
        cv.VideoWriter: An OpenCV VideoWriter object for writing video frames.
    """
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # 'mp4v' is commonly used for MP4 files
    # Define the output filename
    filename = "A_star_Indraneel_Mulakaloori.mp4"
    # Define the frames per second (fps) for the video
    fps = 30
    # Initialize the VideoWriter object with the filename, codec, fps, and resolution
    video_output = cv.VideoWriter(filename, fourcc, fps, (WIDTH, HEIGHT))
    return video_output  

def main():
    print("Program Start")
    print("Please enter the start and end coordinates.")
    print("Coordinates should be given as integers in units of mm from the bottom left origin.")
    print("Image WIDTH is 600 mm. Image HEIGHT is 250 mm.")

    # The scale factor is the resolution of the image for pathing. A scale factor of 2
    # makes the image .5 mm in resolution. DO NOT MODIFY. 
    

    # Generate and expand the obstacle map.
    obstacle_map = gen_obstacle_map(SCALE_FACTOR)
    # Prompt the user for robot radius, clearance and wheel RPMs.
    
    # Wheel Radius (R): 38 mm 
    # Robot Radius (r): 220 mm
    # Wheel Distance (L): 287 mm
    
    ### Robot Radius (R): 220 mm but for scaling purposes, downscaled to 1 pixel
    robot_radius = 1 
    clearance = int(input(f"Enter the clearance radius (int): "))

    # Expand the obstacle space for the robot radius then for the clearance. 
    expanded_obstacle_map, obs_map_gray = expand_obstacles(obstacle_map, SCALE_FACTOR, robot_radius) ### Robot Radius
    expanded_obstacle_map2, obs_map_gray = expand_obstacles(expanded_obstacle_map, SCALE_FACTOR, clearance) ### Clearance
    
    # Prompt the user for the start and end points for planning.
    start_x, start_y, start_theta = get_point(prompt="start", SCALE_FACTOR=SCALE_FACTOR, bloated_map=expanded_obstacle_map2)
    end_x, end_y = get_point(prompt="end", SCALE_FACTOR=SCALE_FACTOR, bloated_map=expanded_obstacle_map2)  
    start_theta = -start_theta % 360    
    video_output = create_video()
    wheel_rpms = get_valid_rpms()
    print("Planning Path...")
    result = A_star((start_x, start_y, start_theta),(end_x,end_y),wheel_rpms,expanded_obstacle_map2,video_output=video_output)
    if result != -1:
        print("Path Found")
        backtrack_list = backtrack(result[0],result[1],[])
        for node in backtrack_list:
            cv.circle(expanded_obstacle_map2, (int(node[1]), int(node[0])), 3, PINK, -1)
        for _ in range(30):
            video_output.write(expanded_obstacle_map2)
    cv.imshow("Map",expanded_obstacle_map2)
    cv.waitKey(0)
    video_output.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()