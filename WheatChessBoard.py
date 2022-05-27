import numpy as np
import matplotlib.pyplot as plt

n_squares = 4
small_board_list = [1]

n_board = 8
m_board = 8

for _ in range(n_squares-1):
    small_board_list.append(2*small_board_list[-1])

print("arrange wheat on a plate of 4 squares (list): {}".format(small_board_list))

small_board_ndarray = np.array(small_board_list)
print("arrange wheat on a plate of 4 squares（ndarray）：{}".format(small_board_ndarray))

########### Problem 1 ####################
#  Number of wheat on a 2x2 chess board
two_two_board_ndarray = small_board_ndarray.reshape(2,2)
#print(two_two_board_ndarray)

########### Problem 2 ####################
# Expansion to n × m mass
#%%timeit
def calculate_wheat_chess_board(n_board, m_board):
    """
    a function that returns an ndarray that describes the number of wheat on the chess board of n × m squares
    using np.append method
    Parameters 
    -----------------------
    n_board : int 
        lenght of the board
    m_board : int 
        width of the board
    
    Returns
    -----------------------
    board_array: ndarray
        wheats on the board array
    """
    n_squares = n_board * m_board
    board_list = [1]

    for _ in range(n_squares-1):
        board_list.append(2*board_list[-1])
    
    board_array = np.array(board_list)
    board_array = board_array.reshape(n_board, m_board)
    return board_array

board_array = calculate_wheat_chess_board(n_board,m_board) 
n_wheats = np.sum(board_array)
print("LIST: number of wheats on the {} x {} board: {}".format(n_board, m_board, n_wheats))
#%%
################## Problem 3 #################
# Total number of wheat
column_array = (board_array.sum(axis=0))/n_board
#print(column_array)

plt.xlabel("column")
plt.ylabel("number")
plt.title("number in each column")
plt.bar(np.arange(1,n_board+1), column_array)
plt.show()

############## Problem 4 ################
# Heat map of the number of wheat

plt.xlabel("column")
plt.ylabel("row")
plt.title("heatmap")
plt.pcolor(board_array)
plt.show()

############## Problem 5 ################
# How many times the second half is the first half

wheat_first_half = np.sum(board_array[0:3,:])
wheat_second_half = np.sum(board_array[4:7,:])

print("number of wheat in the first half: {}".format(wheat_first_half))
print("number of wheat in the second half: {}".format(wheat_second_half))
print("ratio of second and first half {}".format(wheat_second_half/wheat_first_half))

############# Problem 6 #################
#%%timeit
def calculate_wheat_chess_boardcast(n_board, m_board):
    """
    a function that returns an ndarray that describes the number of wheat on the chess board of n × m squares
    using broadcast method

    Parameters 
    -----------------------
    n_board : int 
        lenght of the board
    m_board : int 
        width of the board
    
    Returns
    -----------------------
    board_array: ndarray
        wheats on the board array
    """
    n_squares = n_board * m_board
    
    indices_of_squares = np.arange(n_squares).astype(np.uint64)
    board_ndarray = 2**indices_of_squares
    #board_array = np.array(board_list)
    board_array = board_ndarray.reshape(n_board, m_board)
    return board_array

board_array = calculate_wheat_chess_boardcast(n_board,m_board) 
n_wheats = np.sum(board_array)
print("number of wheats on the {} x {} board: {}".format(n_board, m_board, n_wheats))

################
#%%timeit
def calculate_wheat_chess_ndarray(n_board, m_board):
    """
    a function that returns an ndarray that describes the number of wheat on the chess board of n × m squares
    using np.append array

    Parameters 
    -----------------------
    n_board : int 
        lenght of the board
    m_board : int 
        width of the board
    
    Returns
    -----------------------
    board_array: ndarray
        wheats on the board array
    """
    n_squares = n_board * m_board
    board_ndarray = np.array([1]).astype(np.uint64)
    print("board ndarray:", board_ndarray)
    for _ in range(n_squares - 1):
      board_ndarray = np.append(board_ndarray, 2*board_ndarray[-1])
    print(board_ndarray)
    board_array = board_ndarray.reshape(n_board, m_board)
    
    return board_array


board_array = calculate_wheat_chess_ndarray(n_board,m_board) 
print(board_array)
n_wheats = np.sum(board_array)
print("number of wheats on the {} x {} board: {}".format(n_board, m_board, n_wheats))