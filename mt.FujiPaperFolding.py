import matplotlib.pyplot as plt
import math


paper_thickness = 0.00008

#formula for thickness of folded paper
# thickness = paper_thickness * 2**n

mt_fuji_height = 3776  #meters
centauri_distance = 4.0175 * 10**16 #meters

times_to_fold = math.log(mt_fuji_height/paper_thickness,2)

print('number of times to fold the paper to reach Mt.Fuji height: {} times'.format(times_to_fold))

def calculate_times_to_fold(height, paper_thickness = 0.00008):
    '''
    This function calculate the number of times to fold the paper to reach desired height

    Parameters
    -----------------------
    height : int
        required height to fold paper

    Returns
    -----------------------
    times_to_fold: int
        number of paper folds to reach required height
    '''
    times_to_fold = math.log(height/paper_thickness,2)
    return times_to_fold



def calculate_times_to_fold_long_paper(height, paper_thickness = 0.00008):
    '''
    This function calculate the paper length to reach desired heigth

    Parameters
    -----------------------
    height : int
        height to calculate required paper lenght

    Returns
    -----------------------
    l: int
        required length of paper
    '''
    n_times = calculate_times_to_fold(height)
    l= ((math.pi*paper_thickness)/6)*(2**n_times + 4)*(2**n_times-1)
    return l

l=calculate_times_to_fold_long_paper(centauri_distance)

print('required paper length: {} m'.format(l))

