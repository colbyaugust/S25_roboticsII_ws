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
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create a subscriber to the detected object pose
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.045, self.timer_update)
    
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
	  
            transform = self.tf_buffer.lookup_transform('base_footprint', odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])

            #correction to avoid tracking termination
            if self.obs_pose is not None and self.goal_pose is not None: 
                obstacle_pose = robot_world_R@self.obs_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
                goal_pose = robot_world_R@self.goal_pose+np.array([robot_world_x,robot_world_y,robot_world_z])
            else:
                obstacle_pose = np.zeros(2)
                goal_pose = np.zeros(2)

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
        else:
        # Get the current object pose in the robot base_footprint frame
            #current_obs_pose, current_goal_pose = self.get_current_poses()
        
            # TODO: get the control velocity command
            cmd_vel = self.controller()

            # publish the control command
            self.pub_control_cmd.publish(cmd_vel)
        #################################################
    
    def controller(self):
        # Instructions: You can implement your own control algorithm here
        # feel free to modify the code structure, add more parameters, more input variables for the function, etc.
        
        ########### Write your code here ###########

        '''
        obs_pose, goal_pose = self.get_current_poses()

        goal_vec = goal_pose[:2]
        obs_vec = obs_pose[:2]

        dist_goal = np.linalg.norm(goal_vec) 
        dist_obs = np.linalg.norm(obs_vec) 
        cmd_vel = Twist()
        
        if dist_goal >= 0.5:
            cmd_vel.linear.x = 0.2 * dist_goal ** 2.0
        else:
            cmd_vel.linear.x = 0.0

        # TODO: Update the control velocity command
        cmd_vel.linear.y = 0.0
        cmd_vel.angular.z = 0.0

        return cmd_vel
        '''

        # TODO: Update the control velocity command

        
        #initializing gains
        att_gain = 1.25
        rep_gain = 0.475

        #defining obstacle repuslive radius
        obj_rep_rad = 0.5

        cmd_vel = Twist()
        cmd_vel.linear.x = 0.0
        cmd_vel.linear.y = 0.0
        cmd_vel.angular.z = 0.0

        #getting current poses of the objects in the image frame
        obstacle_pose, goal_pose = self.get_current_poses()

        #computing distances from robot to objects
        goal_vec = goal_pose[:2]
        obs_vec = obstacle_pose[:2]
        dist_to_goal = np.linalg.norm(goal_vec)
        dist_to_obs = np.linalg.norm(obs_vec)

        #computing attractive force
        if dist_to_goal > 0.3: 
            att_force = att_gain * goal_vec / dist_to_goal
        else:
            att_force = np.zeros(2)

        #computing repulsive force
        if dist_to_obs < obj_rep_rad and dist_to_obs > 0.001:
            rep_force = rep_gain * (1.0 / dist_to_obs - 1.0 / obj_rep_rad) / (dist_to_obs ** 2) * (obs_vec / dist_to_obs)
        else:
            rep_force = np.zeros(2)

        #computring net forces
        total_force = att_force - rep_force

        #updating linear velocities clipping to reduce max speed
        cmd_vel.linear.x = np.clip(total_force[0], -0.15, 0.15)
        cmd_vel.linear.y = np.clip(total_force[1], -0.05, 0.05)

        #computing the desired angular heading for the linear velocities
        net_angle = math.atan2(total_force[1], total_force[0])

        #using the net force angle aligns orientation with the actual movement direction to updated robot rotation
        ang_gain = 1.0
        cmd_vel.angular.z = np.clip(ang_gain * net_angle, -0.075, 0.075)

        #returning computed velocity
        return cmd_vel

 
    
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
