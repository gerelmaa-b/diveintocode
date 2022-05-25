import matplotlib.pyplot as plt

total_bowl_rice = 0
bowl_rice = 1
days = 100
 
total_bowl_rice_list = []
day_bowl_rice_list = []
total_bowl_rice = bowl_rice
that_day_bowl_rice = bowl_rice

'''
for i in range(days):
    total_bowl_rice_list.append(total_bowl_rice)
    day_bowl_rice_list.append(that_day_bowl_rice)
    that_day_bowl_rice = total_bowl_rice * 2
    total_bowl_rice = total_bowl_rice + that_day_bowl_rice
'''

def compute_sorori_shinzaemon(days = 30, start_day = 1):
    """
    Parameters
    --------------------
    days : int
        Number of days to get rice (default : 100)
    
    Returns
    --------------------
    function is return value for following order:
        1. total_bowl_rice_list : list
            List of the number of rice you get one day (list length equals n_days)
        2. that_day_bowl_rice : list
            List of the number of rice you get one day (list length equals n_days)
    """

    total_bowl_rice_list=[]
    day_bowl_rice_list=[]
    that_day_bowl_rice = start_day
    total_bowl_rice = start_day

    for i in range(days):
        total_bowl_rice_list.append(total_bowl_rice)
        day_bowl_rice_list.append(that_day_bowl_rice)
        that_day_bowl_rice = total_bowl_rice * 2
        total_bowl_rice = total_bowl_rice + that_day_bowl_rice
    return total_bowl_rice_list, day_bowl_rice_list

def compute_days_to_live(n_rice, n_people=1):
    """
    Parameters
    ----------------------
    n_rice : int
        number of rice grains
    n_people : int
        number of people (default : 1)
    
    Returns
    ----------------------
    n_days : int
        number of days to live
    """
    n_days = n_rice/3*n_people
    return n_days


total_bowl_rice_list, day_bowl_rice_list = compute_sorori_shinzaemon(10)
n_days = compute_days_to_live(total_bowl_rice_list[-1], 10)
print("number of days to live: {} days".format(n_days), "number of months to live: {} months".format(n_days/30))
fig, ax = plt.subplots()
ax.plot(total_bowl_rice_list, color='red', label = 'the total number of rice you receive by that day')
ax.plot(day_bowl_rice_list, color='green', label ='the number of rice you receive on that day')
plt.title("The number of rice")
plt.xlabel("number of days")
plt.ylabel("number of rice")
#plt.tick_params(labelsize=14) # Make settings related to axis values
#plt.plot(folded_thickness_list)
plt.legend()
plt.show()


print("total bowl rice: {} bowls".format(total_bowl_rice))