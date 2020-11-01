import numpy as np

BOARD_SIZE = 8

def initialize_values():
    values = \
        [[100, -20, 10, 5, 5, 10, -20, 100], 
         [-20, -50, -2, -2, -2, -2, -50, -20], 
         [10, -2, -1, -1, -1, -1, -2, 10], 
         [5, -2, -1, -1, -1, -1, -2, 5], 
         [5, -2, -1, -1, -1, -1, -2, 5], 
         [10, -2, -1, -1, -1, -1, -2, 10],  
         [-20, -50, -2, -2, -2, -2, -50, -20],  
         [100, -20, 10, 5, 5, 10, -20, 100]]
    return np.array(values)

def check_mobility(pid, i, j, board_state):
    m_p, m_o = 0, 0
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            if board_state.isValidMove(pid, i, j):
                m_p += 1
            if board_state.isValidMove(-1 * pid, i, j):
                m_o += 1