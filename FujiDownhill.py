import numpy as np 
import matplotlib.pyplot as plt 

csv_path = "mtfuji_data.csv" 

np.set_printoptions(suppress=True) 
fuji = np.loadtxt(csv_path, delimiter=",", skiprows=1)

#"the number of the point", "latitude", "longitude", "elevation", and "distance from point 0"
############# Problem 1 ################## 
# Data visualization

elevation = fuji[:,3]
position = fuji[:,0]

#plt.plot(position, elevation)
#plt.xlabel('position')
#plt.ylabel('elevation [m]')
#plt.title('Mt. Fuji')
#plt.show()

############ Problem 2 ##################
# Create a function to calculate the gradient at a certain point

def compute_gradient(elevation, current_pos, x_change):
    '''
    Calculate the gradient using the amount of change.
    Parameters
    -------------------------------
    current_pos: ndarray
       the number of the current point.
    next_pos: ndarray
       the number of the next point you are now (number of the current point-1)

     Returns
     --------------------------------
    gradient: float or int
       One element is reduced to make a difference
    '''

    y_change = elevation[current_pos] - elevation[current_pos-1]
    gradient = y_change/x_change
    return gradient

#################### Problem 3 ###########################
# Create a function to calculate the destination point

def compute_destination_point(current_position, current_point_grad, learning_rate = 0.2):
    '''
    Calculate the next point to move to based on the information on the slope of the current point.
    Parameters
    -------------------------------
    current_position: int
       current_position
     learning_rate: floating point
        descending changes

     Returns
     --------------------------------
     destination_point: int
        calculated destination point
    '''
    destination_point = int(current_position - learning_rate * current_point_grad)
    return destination_point

################# Problem 4, 5 #############################
# Create a function to go down the mountain
# Visualization of the descent process

def compute_path(elevation, current_pos, position, learning_rate = 0.2):
    '''
    Calculate go down the mountain

    Parameters
    -------------------------------
    current_position: int
       current_position
     learning_rate: floating point
        descending changes

     Returns
     --------------------------------
     destination_point: int
        calculated destination point
    '''
    x_step = position[1]-position[0]
    grad = elevation[current_pos]-elevation[current_pos-1]
    update_y = [elevation[current_pos]]
    update_point = [current_pos]
    fig, ax = plt.subplots()
    ax.plot(position, elevation, color='red')
    while grad > 0:
        grad = compute_gradient(elevation, current_pos, x_step)
        current_pos = compute_destination_point(current_pos, grad, learning_rate)
        update_y.append(elevation[current_pos])
        update_point.append(current_pos)   
        ax.scatter(current_pos, elevation[current_pos], s=15)
        ax.plot([update_point[-1],update_point[-2]], [update_y[-1],update_y[-2]], color = 'orange')
        plt.pause(1)
    plt.show()
    return update_y, update_point

path = []
path_index = []
current_location = 136
learning_rate = 0.2

path, path_index = compute_path(elevation, current_location, position, learning_rate)
fig, ax = plt.subplots()
ax.plot(position, elevation, color='red')
ax.scatter(path_index, path, s=15)
plt.show()

############## Problem 6, 7 ##################
# Change of initial value


path = []
path_index = []
current_location = 120


path, path_index = compute_path(elevation, current_location, position)
fig, ax = plt.subplots()
ax.plot(position, elevation, color='red')
ax.scatter(path_index, path, s=15)
plt.show()

############ Problem 8 ################
# Changing hyperparameter

path = []
path_index = []
current_location = 136
learning_rate = 0.1

path, path_index = compute_path(elevation, current_location, position, learning_rate)
fig, ax = plt.subplots()
ax.plot(position, elevation, color='red')
ax.scatter(path_index, path, s=15)
plt.show()