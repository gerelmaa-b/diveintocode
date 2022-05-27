import matplotlib.pyplot as plt
import math

'''
Assumptions:

    -medium sized chestnut diameter is 3.175cm
    -solar system radius is 39.5 * 1.496*10**8 m
    -Assumed chestnuts is a round object. Therefore, the equation calculate volume of sphere = (4/3)*pi*r**3 (m^3)
    -packing density of spheres is about pi/3*sqrt(2) = 0.740486 

'''

r=0.03175/2 
chestnut_volume = (4/3)*math.pi*r**3
system_rad = 39.5 * 1.496*10**8
system_volume = (4/3)*math.pi*system_rad**3

doubled_time = 5

total_cover_time_list = []
cover_time_list = []
times=[]

print('chestnuts volume:{} m^3'.format(chestnut_volume))
print('solar system volume: {} m^3'.format(system_volume))

def calculate_time_to_cover(system_volume, obj_volume, doubled_time=5):
    '''
    packing density of spheres - pi/3*sqrt(2) is about 0.740486
    To calculate number of chestnuts to fill solar system volume, I used following equation.

        number of chestnutes = 0.740486 * system_volume / chesnuts volume

    Parameters
    -----------------------------
    system_volume : int
        system volume that is covered by object (meters^3)
    
    obj_volume : int
        object volume that cover the system (meters^3)

    Returns
    ------------------------------
    total_cover_time_list : list
        List of total number of chestnuts you get that time 
    cover_time_list : list
        List of the number of chestnuts you every 5 minutes
    ------------------------------
    '''
    i = 1
    time=0
    times = []
    cover_time_list = []
    total_cover_time_list = []
    calc_volume = obj_volume
    total_volume=obj_volume

    n_object = 0.740486 * system_volume / obj_volume
    while i < n_object:
        calc_volume = 2 * calc_volume
        total_volume = total_volume + calc_volume
        i = i * 2
        time= time + doubled_time
        total_cover_time_list.append(total_volume)
        cover_time_list.append(calc_volume)
        times.append(time) 
    
    print('{} chestnuts fill the solar system'.format(n_object))

    return cover_time_list, total_cover_time_list, times


cover_time_list, total_cover_time_list, times = calculate_time_to_cover(system_volume, chestnut_volume)

print("total time to cover system: {} minutes".format(times[-1]))


fig, ax = plt.subplots()
ax.plot(times,total_cover_time_list, color='red', label = 'the total volume')
ax.plot(times,cover_time_list, color='green', label ='that time volume by added')
plt.title("Volume of chestnuts")
plt.xlabel("time [min]")
plt.ylabel("volume [m^3]")
plt.legend()
plt.show()




