# ENPM661_Project_3_phase_2

- Aidan Stark (UID : 113907074)
- Masum Thakkar (UID : 121229076)
- Indraneel Mulakaloori (UID : 121377715)



**Libraries / Dependencies Used**:
- numpy
- cv2
- heapq
- time
- functools
- rclpy
- geometry_msgs
- pathlib

**Video Output Explanation**:
- The video shows the node exploration and the generation of the optimal path. 
- The gray lines show the node exploration. 
The magenta circle is where the program starts and the cyan circle is the end goal. Gray curves (approximated to straight lines in `OPENCV`) extends as the differential drive of mobile robot. 
- The robot performs A* search and finds the optimal path where the cost of each node is computed by `Euler Integration`.
- After generating the optimal path, the program visualise the backtracking path and stores the wheel RPM's in a text file for ROS publisher. The terminal shows `Path Found` and starts to publish the RPM's of the mobile robot for simulation.
- The ROS publisher in the program, reads the control velocities that are stored previously, and publishes them.

 

**Color Key**:
- Black = Unexplored Free Space
- Dark Gray = Explored Free Space
- Red = Obstacle Space
- Orange = Clearance Space Around Obstacles and Walls
- Magenta = Start Point
- Cyan = End Point
- Magenta Line = Final Path

**How To Operate Program**:
`Measuremnts in centimeters`
1. Run the program via the terminal.
2. The terminal will prompt the user for the clearance radius in mm. **Suggested values: 90**
3. The terminal will prompt the user for the Wheel RPMS. This is how we control the robot by providing the left and right wheels. **Enter the first radius/sec value for the motors (int) (1-10 suggested). Suggested value : 3,6**
4. The terminal will prompt the user for the start and end coordinates. 
    1) It will ask for the coordinates as integers, in units of cm, and with respect
    to the center left side as the origin. **Please note that the box is 600 long
    in x and and 300 long in y**. 
    2) If the point given is invalid, it will reprompt the user. It will also ask
    the user for the start orientation for the robot. It will require that the orientation given
    is between 0-359. 
    **Suggested Test Point:** 
     ``
     start_x = 0, start_y = 0, start_theta = 0
     end_x = 585, end_y = 0
    `` 
    
5. The program will then output **Planning Path...**. Wait until the program outputs the final path **Goal Reached**. It will also return the time taken for completion.
6. When **Program Finished** is output to the terminal, it will either display the final path to the screen and generate a video file of the generated
solution (part 1), or begin to publish the solution as a ROS node to the turtlebot in Gazebo. 

**Expected Time to Execute Program**: 
The longest paths that span the whole map take `~3 minutes`. 

Part 01:
Make sure you are in the same directory as the file "python a_star_phase2_FINAL.py" within the terminal.
run the program with "python a_star_phase2_FINAL.py" in the terminal.
Follow instructions in "How to Operate Program".

Part 02: 
Colcon build and source your workspace.
launch the world in Gazebo with "ros2 launch turtlebot3_project3 competition_world.launch.py"
launch our ROS node in a seperate terminal with "ros2 run turtlebot3_project3 path_plan_follow.py"
From here, follow the steps as prompted in "How to Operate Program".

 
Link To Code on GitHub: https://github.com/IndraNeelMulakaloori/ENPM661_Project_3_phase_2.git


2D Path Plan video: https://youtu.be/yOOYC23qgDg 

Gazebo Path Plan Video: https://youtu.be/B5i-X3uU_io 
