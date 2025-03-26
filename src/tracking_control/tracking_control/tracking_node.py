import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    """
    Returns a 3 x 3 cross product matrix for a 3 x 1 vector

             [  0 -k3  k2]
     khat =  [ k3   0 -k1]
             [-k2  k1   0]

    :type    k: numpy.array
    :param   k: 3 x 1 vector
    :rtype:  numpy.array
    :return: the 3 x 3 cross product matrix
    """

    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    """
    Converts a quaternion into a 3 x 3 rotation matrix according to the
    Euler-Rodrigues formula.
    
    :type    q: numpy.array
    :param   q: 4 x 1 vector representation of a quaternion q = [q0;qv]
    :rtype:  numpy.array
    :return: the 3x3 rotation matrix    
    """
    
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2
######################

def euler_from_quaternion(q):
    w=q[0]
    x=q[1]
    y=q[2]
    z=q[3]
    # euler from quaternion
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    return [roll,pitch,yaw]

class TrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # Current object pose
        self.obs_pose = None
        self.goal_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        ###creating publisher for control velo to be sent out
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        ###here is the subscription to get object & goal position###
        # Create a subscriber to the detected object pose
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)
        # Create timer, running at 100Hz, maybe make slower to start and slowly ramp up with preformance
        self.timer = self.create_timer(0.01, self.timer_update)
    
    def detected_obs_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        #self.get_logger().info('Received Detected Object Pose')
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # TODO: Filtering
        # You can decide to filter the detected object pose here
        # For example, you can filter the pose based on the distance from the camera
        # or the height of the object
        # if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
        #     return
        
        try:
            # Transform the center point from the camera frame to the world frame
            transform = self.tf_buffer.lookup_transform(odom_id,msg.header.frame_id,rclpy.time.Time(),rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w,transform.transform.rotation.x,transform.transform.rotation.y,transform.transform.rotation.z]))
            cp_world = t_R@center_points+np.array([transform.transform.translation.x,transform.transform.translation.y,transform.transform.translation.z])
        except TransformException as e:
            self.get_logger().error('Transform Error: {}'.format(e))
            return
        
        # Get the detected object pose in the world frame
        self.goal_pose = cp_world
        
    def get_current_poses(self):
        
        odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value
        # Get the current robot pose
        try:
            # from base_footprint to odom
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            obstacle_pose = robot_world_R@self.obs_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
            goal_pose = robot_world_R@self.goal_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
        
        except TransformException as e:
            self.get_logger().error('Transform error: ' + str(e))
            return
        
        return obstacle_pose, goal_pose
    
    def timer_update(self):
        ################### Write your code here ###################
        
        # Now, the robot stops if the object is not detected
        # But, you may want to think about what to do in this case
        # and update the command velocity accordingly
        if self.goal_pose is None:
            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = 0.0
            self.pub_control_cmd.publish(cmd_vel)
            return
        
        # Get the current object pose in the robot base_footprint frame
        current_obs_pose, current_goal_pose = self.get_current_poses()
        
        # TODO: get the control velocity command
        cmd_vel = self.controller()
        
        # publish the control command
        self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    def controller(self):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        
        ########### Write your code here ###########

        #okay I can edit it now
        #so based off what we did in class x is foward velo and y is side to side
        #outputs are in x, y distance from the robot camera to the object
        #how do I get these positions/access them...gonna want to read through all this stuff but plan is to use the potential field method
        #goal is to get the robot to stop 0.3m from the basket, idk if the robot is good enough to detect from that far
        #so adding topic to rviz would be opening rviz and then pulling the topic like in lab 2
        #all the work computing the distances between the objects is already done...
        #do I need to call the function to get the info...

        #so the idea is the final position of the obstical once it leaves the image frame will
        #be held and enventually decay into nothing
    
        #subscribed topics: 
        #obstacle pose obs_pose
        #goal pose goal_pose
        
        #intilizing gains
        att_gain = 1
        rep_gain = 1

        #object repulsive radius
        obj_rep_rad = 0.5

        cmd_vel = Twist()

        if self.goal_pose is None or self.obs_pose is None:
            return cmd_vel

        try:
        # Get poses in the robot's frame using get current poses
        obstacle_pose, goal_pose = self.get_current_poses()

        # Compute vectors from robot to goal and obstacle 
        #creating vectors from robot to goal and obstacle
        goal_vec = goal_pose[:2]
        obs_vec = obstacle_pose[:2]

        # Distance to goal and obstacle
        # computing distance magnitude using np.linalg.norm (euclie
        dist_to_goal = np.linalg.norm(goal_vec)
        dist_to_obs = np.linalg.norm(obs_vec)

        # Attractive force (pull toward goal)
        if dist_to_goal > 0.3:  # Stop 0.3m away from the goal
            att_force = att_gain * goal_vec / dist_to_goal
        else:
            #sets velo to zero if within the distance 
            att_force = np.zeros(2)

        # Repulsive force (push away from obstacle) ensuring the distance to obstical is positive (idk if nessisary) 
        if dist_to_obs < obj_rep_rad and dist_to_obs > 0.001:
            rep_force = rep_gain * (1.0 / dist_to_obs - 1.0 / obj_rep_rad) / (dist_to_obs ** 2) * (obs_vec / dist_to_obs)
        else:
            #is outside the radius of repulsion is set to 0
            rep_force = np.zeros(2)

        # Net force
        total_force = att_force - rep_force

        # Set linear velocities based on net force
        #np.clip(value, min, max)
        cmd_vel.linear.x = np.clip(total_force[0], -0.5, 0.5)
        cmd_vel.linear.y = np.clip(total_force[1], -0.5, 0.5)

        # Optional: Angular correction to face goal
        goal_angle = math.atan2(goal_vec[1], goal_vec[0])
        cmd_vel.angular.z = np.clip(2.0 * goal_angle, -1.0, 1.0)

        except Exception as e:
            self.get_logger().warn(f"Controller error: {e}")

        return cmd_vel

    
        # TODO: Update the control velocity command
        #cmd_vel = Twist()
        #cmd_vel.linear.x = 0
        #cmd_vel.linear.y = 0
        #cmd_vel.angular.z = 0
        #return cmd_vel
    
        ############################################

def main(args=None):
    # Initialize the rclpy library
    rclpy.init(args=args)
    # Create the node
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    # Destroy the node explicitly
    tracking_node.destroy_node()
    # Shutdown the ROS client library for Python
    rclpy.shutdown()

if __name__ == '__main__':
    main()
