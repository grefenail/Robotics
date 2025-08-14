import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

#plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    #calculate and plot covariance ellipse
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0],estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def g(ut, mu_t_minus_1):
    x, y, theta = mu_t_minus_1
    v, w = ut
    dt = 0.1

    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + w * dt
    theta_new = normalize_angle(theta_new)  # Normalize angle

    return np.array([x_new, y_new, theta_new])

def G_t(ut, mu_t_minus_1):
    
    x, y, theta = mu_t_minus_1  # State: [x, y, theta]
    v, w = ut  # Control inputs: [linear_velocity, angular_velocity]
    dt = 0.1  # Time step

    # Jacobian matrix of g
    G = np.array([
        [1, 0, -v * np.sin(theta) * dt],
        [0, 1,  v * np.cos(theta) * dt],
        [0, 0, 1]
    ])
    return G

# def h(mu_t):
#     # Example: Sensor measures distance and bearing
#     x, y, theta = mu_t  # State: [x, y, theta]
#     landmark = np.array([5.0, 5.0])  # Landmark position

#     dx = landmark[0] - x
#     dy = landmark[1] - y

#     # Distance and bearing to the landmark
#     distance = np.sqrt(dx**2 + dy**2)
#     bearing = np.arctan2(dy, dx) - theta

#     return np.array([distance, bearing])

def H_t(mu_t):
    x, y, theta = mu_t  # State: [x, y, theta]
    landmark = np.array([5.0, 5.0])  # Landmark position

    dx = landmark[0] - x
    dy = landmark[1] - y
    q = dx**2 + dy**2  # Squared distance

    # Jacobian matrix of h
    H = np.array([
        [-dx / np.sqrt(q), -dy / np.sqrt(q), 0],
        [ dy / q,          -dx / q,         -1]
    ])
    return H

def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    '''***        ***'''
     # Current control inputs [v, w] (linear velocity and angular velocity)
    control_inputs = [delta_trans, delta_rot1 + delta_rot2]

    # Update the mean (state transition)
    mu = g(control_inputs, mu)

    # Compute the Jacobian of g
    G = G_t(control_inputs, mu)

    # Define motion noise
    R = np.array([
        [0.2, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.02]
    ])

    # Update the covariance
    sigma = np.dot(G, np.dot(sigma, G.T)) + R

    return mu, sigma

def correction_step(sensor_data, mu, sigma, landmarks):
    # updates the belief, i.e., mu and sigma, according to the
    # sensor model
    # 
    # The employed sensor model is range-only
    #
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 

    x = mu[0]
    y = mu[1]
    theta = mu[2]

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    '''your code here'''
    '''***        ***'''

    R = 0.5 * np.eye(len(ids))
    for i, landmark_id in enumerate(ids):
        # Landmark position
        lm_x, lm_y = landmarks[landmark_id]

        # Difference between the robot and landmark positions
        delta_x = lm_x - x
        delta_y = lm_y - y

        # Squared distance
        distance_sq = delta_x**2 + delta_y**2

        # Measurement Jacobian
        H = np.array([
            [-delta_x / np.sqrt(distance_sq), -delta_y / np.sqrt(distance_sq), 0]
        ])

        # Kalman gain
        K = np.dot(np.dot(sigma, H.T), np.linalg.inv(np.dot(np.dot(H, sigma), H.T) + R[i,i]))

        # Actual measurement
        z = np.array([ranges[i]])

        # Update state estimate
        mu = mu + np.dot(K, (z - np.array([np.sqrt(distance_sq)])))

        # Update covariance
        I = np.eye(len(mu))
        sigma = np.dot(I - np.dot(K, H), sigma)
    return mu, sigma

def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../kf_framework/data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../kf_framework/data/sensor_data.dat")

    #initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],\
                      [0.0, 1.0, 0.0],\
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    #run kalman filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)

        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    plt.show('hold')

if __name__ == "__main__":
    main()
