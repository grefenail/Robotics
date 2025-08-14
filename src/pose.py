import requests
import math
import rospy
from std_msgs.msg import String

# Initialize the ROS node
rospy.init_node('find_target', anonymous=True)

def calculate_adjacent(data):
    split_data = 0
    # Define the API endpoint
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

    if not split_data:
        split_data = (data.data).split(";")
        #split_data = str(split_data)
        start_theta = float(split_data[0])
        print("the start_theta is: " + str(start_theta))
        #depth value
        # Extract the theta value from the data
        opposite = float(split_data[1])
        print("the opposite is: " + str(opposite))

        # Convert angle to radians
        current_rad = float(math.radians(theta1))
        minus_rad = abs(math.radians(start_theta)) - abs(current_rad)
        other_corner = 90 - minus_rad

        # Calculate the adjacent side using the tangent formula
        find_adjacent = (1/math.tan(other_corner)) * opposite
        
        if minus_rad > 0:
            rospy.loginfo("This is positive ")
            rospy.loginfo("This is opposite range: %s", opposite)
            rospy.loginfo("This is find_adjacent range: %s", find_adjacent)
            nx = x1 + abs(opposite)
            ny = y1 - abs(find_adjacent)
            print("doneeeeeeeeeeeeeeeeeeeeee")
            nt = theta1
        else:
            rospy.loginfo("This is negative ")
            rospy.loginfo("This is opposite range: %s", opposite)
            rospy.loginfo("This is find_adjacent range: %s", find_adjacent)
            nx = x1 - abs(opposite)
            ny = y1 - abs(find_adjacent)
            print("doneeeeeeeeeeeeeeeeeeeeee")
            nt = theta1

    # Define the API endpoint
    endpoint2 = "http://192.168.10.2/cmd/nav"

    # Example coordinates
    x2 = nx
    y2 = ny
    theta_degrees = nt

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

# Subscribe to the 'pub_find' topic
sub = rospy.Subscriber('pub_find', String, calculate_adjacent)

# Spin the ROS node
rospy.spin()
