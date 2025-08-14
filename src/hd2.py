import pyrealsense2 as rs
import numpy as np
import cv2
import tensorflow as tf
import rospy , requests
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from math import radians
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# import math functions
from math import *




global twist
global depth_value
depth_value = 0
    # Initialize the ROS node
rospy.init_node('human_detection_node', anonymous=True)

#url = 'http://192.168.10.2/cmd/speed'
#body = {"vx": 0, "vth": 0}

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Disable auto exposure and auto white balance
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Load the SSD model
model_path = '/home/adam/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb'
detection_graph = tf.Graph()

# Create a publisher to publish the image
# image_pub = rospy.Publisher('image_topic', Image, queue_size=1)
# image1_pub = rospy.Publisher('image_topic', Image, queue_size=1)
#pub_rot = rospy.Publisher('pub_rot', String, queue_size=1)
pub_find = rospy.Publisher('pub_find', String, queue_size=1)
pub_move = rospy.Publisher('cmd_vel', Twist, queue_size=1)
twist = Twist()

# Create a CvBridge instance to convert OpenCV images to ROS images
bridge = CvBridge()
xcoor = 0
ycoor = 0

def main():

    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # Create a TensorFlow session and get references to input and output tensors
    with detection_graph.as_default():
        sess = tf.compat.v1.Session(graph=detection_graph)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Define class labels corresponding to the model's output classes
    labels = ["background", "person"]  # Add "person" as the label for humans

    # Start the RealSense pipeline
    pipeline.start(config)

    selected_person = None  # Variable to store the selected person index

    rospy.Rate(0.1)
    # Initialize variables to keep track of the highest positive rate and the corresponding person index
    highest_positive_rate = 0

    #detected = False

    zero_depth_timer_start = None
    zero_depth_threshold = rospy.Duration(5)  # 5 seconds
    current_sec = time.time()

    theta1 = 0
    start_theta = theta1
    count = 0
    human_exist = False
    selected_person = None
    global yellow 
    cur_time = 2000000000
    print("curtime is: " + str(cur_time))
    target_detected = False
    start_timer = False

    # Create a linear regression model
    linear_reg_model = LinearRegression()

    # Lists to store historical data
    history_positions = []
    xarray = np.array([])
    yarray = np.array([])
    # Sliding window size
    window_size = 3
    xcoor = np.array([])
    ycoor = np.array([])
    next_coordinate_x = 0
    next_coordinate_y = 0
    add_x_vals_original = np.array([])
    add_y_vals_original =np.array([])
    add_X_train = np.array([])
    add_y_train = np.array([])
    # X_train = np.array([])
    # y_train = np.array([])
    global color_image
     # initial parameters
    measurement_sig = 4.
    motion_sig = 2.
    mu_x = 0.
    sig_x = 100000.
    mu_y = 0.
    sig_y = 100000.
    try:
        
        while True:
            
            zone = True
            # Wait for frames
            frames = pipeline.wait_for_frames()

            # Get color frame
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the color frame to a numpy array
            color_image = np.asanyarray(color_frame.get_data())

            # Perform object detection on the color image
            with detection_graph.as_default():
                image_np_expanded = np.expand_dims(color_image, axis=0)
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded}
                )

            # if target_detected == False:
                
            #     print("triggered")
            #     control_coordinate_x = int(next_coordinate_x)
            #     control_coordinate_y = int(next_coordinate_y)
            #     coordinate(control_coordinate_x, control_coordinate_y)
            #     #####################################################################

           
           
            if target_detected == False and start_timer == True:
                stloti = time.time()
                #print("stloti is: " + str(stloti))
                # start = True
                
                #i need to make this only run once
                if stloti - cur_time >= 5:
                    print("more than 5 seconds")
                    print("triggered")
                    control_coordinate_x = int(next_coordinate_x)
                    control_coordinate_y = int(next_coordinate_y)
                    coordinate(control_coordinate_x, control_coordinate_y)
                    #####################################################################
                    start_timer = False
                    print("sendddddddddddddddddddddddddddddddddddddd")
                print("target cant be seen")
           
                
            target_detected = False
            # Inside the loop after detecting a person
            for i in range(int(num[0])):
                #print("see everyhting")   
                firstloop = True     
                #print("first loop")
                if scores[0, i] > 0.7:  # You can adjust the confidence threshold here
                    #print("see everyhting") 
                    # print("2222222222222222222222222222")
                    class_id = int(classes[0, i])
                    if class_id < len(labels) and labels[class_id] == "person":
                        #print("person seen") 
                        
                        ymin, xmin, ymax, xmax = boxes[0, i]
                        left = int(xmin * color_image.shape[1])
                        top = int(ymin * color_image.shape[0])
                        right = int(xmax * color_image.shape[1])
                        bottom = int(ymax * color_image.shape[0])
                        human_exist = True
                        #
                        # Define a bounding box for the person
                        person_roi = color_image[top:bottom, left:right]
                      
                        not_detected = False
                        # Convert the person ROI to HSV color space
                        hsv_person = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)


                        # Black for top
                        lower_dark_grey = np.array([0, 0, 0])
                        upper_dark_grey = np.array([180, 255, 20])

                        # Black for bottom
                        lower_black = np.array([0, 0, 0])
                        upper_black = np.array([180, 255, 20])

                        # # white for top
                        # lower_dark_grey = np.array([0, 0, 0])
                        # upper_dark_grey = np.array([0, 0, 255])

                        # # white for bottom
                        # lower_black = np.array([0, 0, 0])
                        # upper_black = np.array([0, 0, 255])

                        # #RGB black bottom and top
                        # lower_dark_grey = np.array([0, 0, 0])
                        # upper_dark_grey = np.array([30, 30, 30])
                        # lower_black = np.array([0, 0, 0])
                        # upper_black = np.array([30, 30, 30])
                        # mask_dark_grey = cv2.inRange(person_roi, lower_dark_grey, upper_dark_grey)
                        # mask_black = cv2.inRange(person_roi, lower_black, upper_black)

                        # # Maroon for top
                        # lower_dark_grey = np.array([0, 120, 100])
                        # upper_dark_grey = np.array([100, 230, 250])

                        

                        # Create masks for dark grey and black regions
                        mask_dark_grey = cv2.inRange(hsv_person, lower_dark_grey, upper_dark_grey)
                        mask_black = cv2.inRange(hsv_person, lower_black, upper_black)

                        positive_rate_top = cv2.countNonZero(mask_dark_grey) / mask_dark_grey.size
                        positive_rate_bottom = cv2.countNonZero(mask_black) / mask_black.size

                        # Check if both dark grey and black are detected
                        #if cv2.countNonZero(mask_dark_grey) > 0 and cv2.countNonZero(mask_black) > 0:
                        # Combine the masks
                        combined_mask = cv2.bitwise_or(mask_dark_grey, mask_black)
                        
                        positive_rate = cv2.countNonZero(combined_mask) % combined_mask.size
                      
                    

                        # if positive_rate == None:
                        #     selected_person = 0

                        if positive_rate > highest_positive_rate:
                            highest_positive_rate = positive_rate
                            selected_person = i

                        if selected_person == i:
                            start_timer = False
                            count = count + 1
                            yellow = True
                            target_detected = True
                            #yellow
                            cv2.rectangle(color_image, (left, top), (right, bottom), (60, 255, 255), 2)
                            # Convert the color_image to a ROS Image message
                            
                            depth_frame = frames.get_depth_frame()
                           
                            depth_image = np.asanyarray(depth_frame.get_data())
                           
                            # Calculate the center of the person's bounding box
                            person_center_x = (left + right) // 2
                            person_center_y = (top + bottom) // 2

                            ###########################################################

                            endpoint1 = "http://192.168.10.2/reeman/pose"

                            # Send a GET request to get the pose data
                            response = requests.get(endpoint1)

                            # Check the response status code
                            if response.status_code == 200:
                                pose_data = response.json()
                                x1 = pose_data.get("x")
                                y1 = pose_data.get("y")
                                theta1 = pose_data.get("theta")
                                rospy.loginfo("Robot's current pose:")
                                rospy.loginfo("X1: %s", x1)
                                rospy.loginfo("Y1: %s", y1)
                                rospy.loginfo("Theta1: %s", theta1)
                            else:
                                rospy.logerr("Failed to retrieve the robot's pose. Check your request or endpoint.")

                            xcoor = -1.69263255, -0.93858029, -0.2455431 
                            ycoor = 1.33423509, 1.38431327, 1.70255854

                            if len(xcoor) > window_size:
                                xcoor = np.delete(xcoor, 0)
                            
                            if len(ycoor) > window_size:
                                ycoor = np.delete(ycoor, 0)

                            if len(xcoor) >= window_size and len(ycoor) >= window_size:
 
                     
                                # Fit the polynomial regression model
                                T_train = np.array(range(len(xcoor))).reshape(-1, 1)
                                X_train = np.array(xcoor).reshape(-1, 1)
                                y_train = np.array(ycoor).reshape(-1, 1)

                                add_X_train = np.append(add_X_train, [X_train])
                                add_y_train = np.append(add_y_train, [y_train])

                                # # Normalize the data
                                # scaler_X = StandardScaler()
                                # scaler_y = StandardScaler()
                                # X_train_scaled = scaler_X.fit_transform(X_train)
                                # y_train_scaled = scaler_y.fit_transform(y_train)

                                # Print the input arrays
                                print("This is the x array:", X_train)
                                print("This is the y array:", y_train)

                                # Create PolynomialFeatures
                                poly_features = PolynomialFeatures(degree=1, include_bias=False)

                                # Transform input features to polynomial features
                                X_poly = poly_features.fit_transform(X_train)
                                y_poly = poly_features.fit_transform(y_train)

                                # Create Polynomial Regression models
                                lin_reg2_x = LinearRegression()
                                lin_reg2_y = LinearRegression()

                                # Train the models
                                lin_reg2_x.fit(T_train, X_poly)
                                lin_reg2_y.fit(T_train, y_poly)

                                next_t = np.array([[len(ycoor)]])
                                print("next_t :" + str(next_t))
                                # Predict the next coordinates
                                next_coordinate_x = lin_reg2_x.predict(next_t)
                                next_coordinate_y = lin_reg2_y.predict(next_t)
                                print('next_coordinate_x: '+ str(next_coordinate_x))
                                print('next_coordinate_y: '+ str(next_coordinate_y))

                                # Plot the original data points
                                plt.scatter(T_train, X_train, color='blue', label='X Coordinate')
                                plt.scatter(T_train, y_train, color='red', label='Y Coordinate')

                                #Plot the linear regression lines
                                plt.plot(T_train, lin_reg2_x.predict(T_train), color='blue', linestyle='dashed', linewidth=2, label='X Regression Line')
                                plt.plot(T_train, lin_reg2_y.predict(T_train), color='red', linestyle='dashed', linewidth=2, label='Y Regression Line')
                                # print("T_train" + str(T_train))
                                # print("poly_features.fit_transform(T_train): " + str(poly_features.fit_transform(T_train)))
                                # confirm = np.array([[len(poly_features.fit_transform(T_train))]])
                                # print("confimr: " + str(confirm))
                                # print("lin_reg2_x.predict(poly_features.fit_transform(T_train)): " + str(lin_reg2_x.predict(confirm)))
                                
                             


                                plt.plot(T_train, lin_reg2_x.predict(poly_features.fit_transform(T_train)),color='red')
                                plt.plot(T_train, lin_reg2_y.predict(poly_features.fit_transform(T_train)),color='blue')

                                # Plot the predicted coordinate point in yellow
                                plt.scatter(next_t, next_coordinate_x, color='yellow', marker='o', label='Predicted X Coordinate')
                                plt.scatter(next_t, next_coordinate_y, color='green', marker='o', label='Predicted Y Coordinate')

                                # Add labels and legend
                                plt.xlabel('Time')
                                plt.ylabel('Coordinates')
                                plt.legend()

                                # Show the plot
                                plt.show()
                                                                

                            control_command_x = person_center_x
                            control_command_y = person_center_y

                            

                          
                            # plotting regression line
                            #plot_regression_line(xarray, yarray, b)
                            #####################################################################
                            print("position x: "+ str(control_command_x))
                            print("position y: "+ str(control_command_y))
                            # Get the depth value at the center of the bounding box
                            depth_value = depth_frame.get_distance(int(control_command_x), int(control_command_y))
                            move(depth_value, control_command_x, control_command_y)
                           
                           
                         
                        else:
                            target_detected = False
                            
                            yellow = False
                            #purple
                            cv2.rectangle(color_image, (left, top), (right, bottom), (180, 30, 255), 2)
            # if start_timer == True:
            #     print("---------------------------------------")     
            # print("start: " + str(start_timer))
            # print("target_detected: " + str(target_detected))
            if target_detected == False and start_timer == False:
                cur_time = time.time() 
                print("curtime is: " + str(cur_time))
                start_timer = True
          
            cv2.imshow('Color Image', color_image)
            #print(count)
            
        

            # Exit the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
 
        # Close the TensorFlow session
        sess.close()
        # Stop the RealSense pipeline and release resources
        pipeline.stop()
        cv2.destroyAllWindows()

def move(depth_value, control_command_x, control_command_y):
    print("moveeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
    #cv2.imshow('Color Image', color_image)
    print(control_command_x, control_command_y)
    # Calculate the center of the frame
    frame_center_x = color_image.shape[1] // 2
    frame_center_y = color_image.shape[0] // 2

    # Draw a crosshair (a cross lining) at the center of the person's bounding box
    crosshair_length = 20  # Adjust the length of the crosshair as needed
    cv2.line(color_image, (control_command_x - crosshair_length, control_command_y), (control_command_x + crosshair_length, control_command_y), (0, 0, 255), 2)
    cv2.line(color_image, (control_command_x, control_command_y - crosshair_length), (control_command_x, control_command_y + crosshair_length), (0, 0, 255), 2)
    # Define the size of the ROI (e.g., 100 pixels in both dimensions)
    roi_size = 200

    # Calculate the new coordinates for the ROI
    roi_x1 = max(control_command_x - roi_size // 2, 0)
    roi_x2 = min(control_command_x + roi_size // 2, color_image.shape[1])
    roi_y1 = max(control_command_y - roi_size // 2, 0)
    roi_y2 = min(control_command_y + roi_size // 2, color_image.shape[0])

    # Draw the ROI rectangle
    roi_color = (0, 0, 255)  # Red color in BGR format (Blue, Green, Red)
    thickness = 2  # Line thickness
    cv2.rectangle(color_image, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, thickness)

    if control_command_x < frame_center_x - 50:
        # Rotate the robot to the right
        rotation_speed = 0.001  # Adjust the rotation speed as needed
        #body["vth"] = 0.0000001
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 25; 
        pub_move.publish(twist)
        cat = "positive"
        # pub_rot.publish(cat)
        #print(cat)
        print("Rotating to the right. Distance from center: " + str(control_command_x) + " pixels")
        
    elif control_command_x > frame_center_x + 50:
        # Rotate the robot to the left
        rotation_speed = 0.001  # Adjust the rotation speed as needed
        #body["vth"] = -0.0000001
        twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = -25; 
        pub_move.publish(twist)
        cat = "negative"
        # pub_rot.publish(cat)
        #print(cat)
        print("Rotating to the left. Distance from center: " + str(-control_command_x) + " pixels")



    if depth_value > 1 and control_command_x < frame_center_x + 50 and control_command_x > frame_center_x - 50:
        #body["vx"] = 0.3
        cat = "forward"
        twist.linear.x = 0.1; twist.linear.y = 0; twist.linear.z = 0
        twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0; 
        pub_move.publish(twist)
        # pub_rot.publish(cat)
        #print(cat)
        print("Robot forward. Distance from robot: " + str(depth_value) + " m")

def coordinate(nx, ny):
    # Define the API endpoint
    endpoint2 = "http://192.168.10.2/cmd/nav"

    # Example coordinates
    x2 = nx
    y2 = ny
    

    # Convert theta to radians
    #theta_radians = math.radians(theta_degrees)

    # Prepare the payload
    payload = {
        "x": x2,
        "y": y2,
        #"theta": theta_degrees
    }

    # Make the POST request
    response = requests.post(endpoint2, json=payload)

    # Check the response status code and content type
    if response.status_code == 200:
        result = response.json()
        if result.get("status") == "success":
            rospy.loginfo("Command sent successfully.")
            rospy.loginfo("X value is: %s", payload["x"])
            rospy.loginfo("Y value is: %s", payload["y"])
            #rospy.loginfo("Theta value is: %s", payload["theta"])
            rospy.signal_shutdown("Terminating after processing depth data")
        else:
            rospy.logwarn("Command failed. Check the result: %s", result)
    else:
        rospy.logerr("Failed to send command. Check the response status code and content type.")


if __name__ == '__main__':
    main()
