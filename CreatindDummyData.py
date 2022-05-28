import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

mean = [-3, 0]
mean_2 = [0, -3]
cov = [[1.0, 0.8],[0.8, 1.0]]


random_data_1 = np.random.multivariate_normal(mean, cov, size=500)
random_data_2 = np.random.multivariate_normal(mean_2, cov, size=500)

########## Problem 2 ###############
# Visualization by scatter plot
print(random_data_1.shape)
plt.scatter(random_data_1[:,0], random_data_1[:,1])
plt.show()

########## Problem 3 ###############
# Visualization by histogram

plt.xlim((-5, 3))
plt.hist(random_data_1[:,0])
plt.show() 
plt.xlim((-5, 3))
plt.hist(random_data_1[:,1])
plt.show() 

########## Problem 4 ###############
# Addition of data

fig, ax = plt.subplots()
ax.scatter(random_data_1[:,0],random_data_1[:,1], color='blue', label = '0')
ax.scatter(random_data_2[:,0],random_data_2[:,1], color='orange', label ='1')
plt.title("Dummy Data")
plt.legend()
plt.show()

######### Problem 5 #################
# Data combination

#using np.concatenate()
combined_data_con = np.concatenate((random_data_1, random_data_2))
print("Combined data by np.concatenate shape: {}".format(combined_data_con.shape))


#using np.vstack()
combined_data_con = np.vstack((random_data_1, random_data_2))
print("Combined data by np.vstack shape: {}".format(combined_data_con.shape))

######### Problem 6 ##################
# labeling

label = np.empty((1000,1))
label[0:500] = 0
label[500:1000] = 1

# count zeros in 1d array
#n_zeros = np.count_nonzero(label==0)
# display the count of zeros
#print(n_zeros)

combined_data_con = np.append(combined_data_con, label, axis = 1)
print(combined_data_con.shape)
print(combined_data_con)

