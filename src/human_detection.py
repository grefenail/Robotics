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

def main():
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

    start_theta = theta1
    count = 0
    human_exist = False
    selected_person = None
    global yellow 
    cur_time = 2000000000
    print("curtime is: " + str(cur_time))
    target_detected = False
    start_timer = False
    
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

            
            #print("zero loop" + str(target_detected))
            if target_detected == False and start_timer == True:
                stloti = time.time()
                
                
                #i need to make this only run once
                if stloti - cur_time >= 5:
                    print("more than 5 seconds")
                    merge_pose = str(start_theta) + ";" + str(depth_value)
                    pub_find.publish(merge_pose)
                    cur_time = time.time()
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


                        # # Black for top
                        # lower_dark_grey = np.array([0, 0, 0])
                        # upper_dark_grey = np.array([180, 255, 20])

                        # # Black for bottom
                        # lower_black = np.array([0, 0, 0])
                        # upper_black = np.array([180, 255, 20])

                        # white for top
                        lower_dark_grey = np.array([0, 0, 0])
                        upper_dark_grey = np.array([0, 0, 255])

                        # white for bottom
                        lower_black = np.array([0, 0, 0])
                        upper_black = np.array([0, 0, 255])

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
                        #cv2.imshow('mask_dark_grey', mask_dark_grey)
                        #cv2.imshow('mask_black', mask_black)
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
                            
                            #pub_find.publish("HIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
                            start_timer = False
                            count = count + 1
                            yellow = True
                            target_detected = True
                            #yellow
                            cv2.rectangle(color_image, (left, top), (right, bottom), (60, 255, 255), 2)
                            # Convert the color_image to a ROS Image message
                            
                            depth_frame = frames.get_depth_frame()
                           
                           

                            # image = np.asarray(color_image)
                            # image1 = np.asarray(depth_frame)
                            depth_image = np.asanyarray(depth_frame.get_data())
                            depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                            #cv2.imshow('depth_color', depth_color)
                          


                       

                            # Calculate the center of the person's bounding box
                            person_center_x = (left + right) // 2
                            person_center_y = (top + bottom) // 2

                            # Get the depth value at the center of the bounding box
                            depth_value = depth_frame.get_distance(person_center_x, person_center_y)
                           
                            #print(depth_value)   
                            # Calculate the center of the frame
                            frame_center_x = color_image.shape[1] // 2
                            frame_center_y = color_image.shape[0] // 2

                            # Draw a crosshair (a cross lining) at the center of the person's bounding box
                            crosshair_length = 20  # Adjust the length of the crosshair as needed
                            cv2.line(color_image, (person_center_x - crosshair_length, person_center_y), (person_center_x + crosshair_length, person_center_y), (0, 0, 255), 2)
                            cv2.line(color_image, (person_center_x, person_center_y - crosshair_length), (person_center_x, person_center_y + crosshair_length), (0, 0, 255), 2)

                         

                            # You can use the relative_x and relative_y values to determine the rotation angle
                            # For example, you can rotate the robot left or right based on the relative_x value
                            rotation_speed = 0.1  # Adjust the rotation speed as needed
                            # Calculate the center of the person's bounding box
                            person_center_x = (left + right) // 2
                            person_center_y = (top + bottom) // 2

                            # Define the size of the ROI (e.g., 100 pixels in both dimensions)
                            roi_size = 200

                            # Calculate the new coordinates for the ROI
                            roi_x1 = max(person_center_x - roi_size // 2, 0)
                            roi_x2 = min(person_center_x + roi_size // 2, color_image.shape[1])
                            roi_y1 = max(person_center_y - roi_size // 2, 0)
                            roi_y2 = min(person_center_y + roi_size // 2, color_image.shape[0])

                            # Draw the ROI rectangle
                            roi_color = (0, 0, 255)  # Red color in BGR format (Blue, Green, Red)
                            thickness = 2  # Line thickness
                            cv2.rectangle(color_image, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, thickness)

                            relative_x = person_center_x - frame_center_x

                            #relative_x = frame_center_x - (person_center_x + roi_size // 2) 

                            #relative_x = (roi_x1 - roi_x2) / 2 - frame_center_x

                            cat = ""
                            

                            if person_center_x < frame_center_x - 50:
                                # Rotate the robot to the right
                                rotation_speed = 0.001  # Adjust the rotation speed as needed
                                #body["vth"] = 0.0000001
                                twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
                                twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 25; 
                                pub_move.publish(twist)
                                cat = "positive"
                                # pub_rot.publish(cat)
                                #print(cat)
                                print("Rotating to the right. Distance from center: " + str(relative_x) + " pixels")
                               
                            elif person_center_x > frame_center_x + 50:
                                # Rotate the robot to the left
                                rotation_speed = 0.001  # Adjust the rotation speed as needed
                                #body["vth"] = -0.0000001
                                twist.linear.x = 0; twist.linear.y = 0; twist.linear.z = 0
                                twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = -25; 
                                pub_move.publish(twist)
                                cat = "negative"
                                # pub_rot.publish(cat)
                                #print(cat)
                                print("Rotating to the left. Distance from center: " + str(-relative_x) + " pixels")
             


                            if depth_value > 1 and person_center_x < frame_center_x + 50 and person_center_x > frame_center_x - 50:
                                #body["vx"] = 0.3
                                cat = "forward"
                                twist.linear.x = 0.1; twist.linear.y = 0; twist.linear.z = 0
                                twist.angular.x = 0; twist.angular.y = 0; twist.angular.z = 0; 
                                pub_move.publish(twist)
                                # pub_rot.publish(cat)
                                #print(cat)
                                print("Robot forward. Distance from robot: " + str(depth_value) + " m")
                         
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
            print(count)
            
        

            # Exit the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Close the TensorFlow session
        sess.close()
        # Stop the RealSense pipeline and release resources
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
