"""
Adapted from https://github.com/jpypi/othello-rl/blob/master/players.py under MIT license.
"""

import numpy as np
import random

import util

BOARD_SIZE = 8
VALUES = util.initialize_values()

class HumanPlayer:
    
    def __init__(self, log_history=True):
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board_state, pid, _):
        try:
            pos = map(int, map(str.strip, input().split(" ")))
            place_func(*pos)
            return True
        except ValueError:
            return False

class RandomAgent:

    def __init__(self, seed=None, log_history=True):
        self.seed = seed
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board_state, pid, _):
        if self.seed:
            random.seed(self.seed)
        # check using the board_state
        possibilities = [[(i,j) for j in range(BOARD_SIZE)] for i in range(BOARD_SIZE)]
        flattened_possibilities = [item for sublist in possibilities for item in sublist]
        while flattened_possibilities:
            pos = random.choice(flattened_possibilities)
            success = place_func(*pos)
            if success:
                return True
            else:
                flattened_possibilities.remove(pos)
        return False

class PositionalAgent:

    def __init__(self, log_history=True):
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board_state, pid, _):
        pct_tiles_left = board_state.remaining_squares / BOARD_SIZE**2
        pos, max_val = None, 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state.isValidMove(pid, i, j):
                    board_state.updateBoard(pid, i, j)
                    if pct_tiles_left < 0.8:
                        val = np.multiply(np.power(board_state.board, 2), VALUES[i,j])
                    else:
                        val = pid * np.sum(board_state.board) 
                    board_state.updateBoard(0, i, j)

                    if val > max_val:
                        pos = (i,j)
                        max_val = val

        try:
            place_func(*pos)
            return True
        except ValueError:
            return False


class MobilityAgent:

    def __init__(self, w1 = 10, w2 = 1, log_history=True):
        self.w1 = w1
        self.w2 = w2
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board_state, pid, _):
        # mobility = # of legal moves a player can make in a certain position
        pct_tiles_left = board_state.remaining_squares / BOARD_SIZE**2
        pos, max_val = None, 0
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board_state.isValidMove(pid, i, j):
                    board_state.updateBoard(pid, i, j)
                    if pct_tiles_left < 0.8:
                        c_p = pid * (board_state.board[0,0] + board_state.board[0,7] + board_state.board[7,0] + board_state.board[7,7])
                        c_o = 4 - c_p
                        m_p, m_o = util.check_mobility(pid, i, j, board_state)
                        val = self.w1 * (c_p - c_o) + self.w2 * (m_p - m_o) / (m_p + m_o)
                    else:
                        val = pid * np.sum(board_state.board)
                    board_state.updateBoard(0, i ,j)
                
                    if val > max_val:
                        pos = (i,j)
                        max_val = val

        try:
            place_func(*pos)
            return True
        except ValueError:
            return False
 


########################################################
# MODEL FREE AGENTS THAT LEARN
"""
class GradientQLearningAgent:

    def __init__(self, alpha, gamma, k_max, explore_strategy):
        self.alpha = alpha
        self.gamma = gamma
        self.k_max = k_max
        self.explore_strategy = explore_strategy
        self.play_history = []

    def play(self, place_func, board_state, pid, log_history=True):
        # convert 2D board state to a one hot encoded state
        s = np.apply_along_axis(lambda x: int((x == pid and 1) or (x != 0 and -1)),
                                1, board_state.reshape((64, 1))).reshape((64, 1))
        pos = self.explore_strategy(place_func)

        if pos == False:
            return False
        elif log_history:
            self.play_history.append((np.copy(s), pos[0]*8 + pos[1]))

    def update_model(self, final_score):
        for i in range(len(self.play_history)):


class ReplayGradientQLearningAgent:
"""
