"""
Adapted from https://github.com/jpypi/othello-rl/blob/master/players.py under MIT license.
"""

import numpy as np
import random
import util
import nn
import conv
import torch
import torch.optim as optim
from torch.autograd import Variable

BOARD_SIZE = 8
VALUES = util.initialize_values()

class HumanPlayer:
    
    def __init__(self, log_history=True):
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board, pid, _):
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

    def play(self, place_func, board, pid, _):
        if self.seed:
            random.seed(self.seed)
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

    def play(self, place_func, board, pid, _):
        pct_tiles_left = board.remaining_squares / BOARD_SIZE**2
        pos, max_val = None, float('-inf')
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board.isValidMove(pid, i, j):
                    temp = board.copy()
                    temp.updateBoard(pid, i, j)
                    board_state = temp.getState()
                    if 1 - pct_tiles_left < 0.8:
                        val = np.sum(np.multiply(
                            pid * np.power(board_state, 2), VALUES[i, j]))
                    else:
                        val = pid * np.sum(board_state)

                    if val > max_val:
                        pos = (i, j)
                        max_val = val

        try:
            place_func(*pos)
            return True
        except TypeError:
            return False


class MobilityAgent:

    def __init__(self, w1 = 10, w2 = 1, log_history=True):
        self.w1 = w1
        self.w2 = w2
        self.play_history = []
        self.wins = 0

    def play(self, place_func, board, pid, _):
        # mobility = # of legal moves a player can make in a certain position
        pct_tiles_left = board.remaining_squares / BOARD_SIZE**2
        pos, max_val = None, float('-inf')
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board.isValidMove(pid, i, j):
                    temp = board.copy()
                    temp.updateBoard(pid, i, j)
                    board_state = temp.getState()
                    if 1 - pct_tiles_left < 0.8:
                        c_p = pid * \
                            (board_state[0, 0] + board_state[0, 7] +
                             board_state[7, 0] + board_state[7, 7])
                        c_o = -1 * pid * \
                            (board_state[0, 0] + board_state[0, 7] +
                             board_state[7, 0] + board_state[7, 7])
                        m_p, m_o = util.check_mobility(pid, i, j, temp)
                        try:
                            val = self.w1 * (c_p - c_o) + self.w2 * (m_p - m_o) / (m_p + m_o)
                        except ZeroDivisionError:
                            continue
                    else:
                        val = pid * np.sum(board_state)
                
                    if val > max_val:
                        pos = (i,j)
                        max_val = val

        try:
            place_func(*pos)
            return True
        except TypeError:
            return False
 


########################################################
# MODEL FREE AGENTS THAT LEARN
# """
class QLearningAgent:

    def __init__(self, alpha=0.07, gamma=0.99, k_max=50, explore_strategy='epsilon_greedy'):
        self.alpha = alpha
        self.gamma = gamma
        self.k_max = k_max
        self.explore_strategy = explore_strategy
        self.nn = nn.QLNN([64**2, 128, 128, 64])
        self.wins = 0
        self.epsilon = 0.6
        self.play_history = []

    def play(self, place_func, board, pid, log_history=True):
        # convert 2D board state to a one hot encoded state
        s = np.apply_along_axis(lambda x: int((x == pid and 1) or (x != 0 and -1)),
                                1, board.board.reshape((64, 1))).reshape((64, 1))
        # pos = self.explore_strategy(place_func)
        # pos = self.nn.forward(s)
        prev_board_state = board.getState()

        actions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board.isValidMove(pid, i, j):
                    temp = board.copy()
                    temp.updateBoard(pid, i, j)
                    board_state = temp.getState()
                    actions.append(board_state.flatten())
                else:
                    # may want to instead pass a zeroed out matrix
                    actions.append(prev_board_state.flatten())

        # (1, 64, 8, 8)
        input_nn = np.array(actions)




        u = self.nn.forward(torch.FloatTensor(input_nn))
        u = u.reshape(1,u.shape[0])
        valid_move = False
        if np.random.random() < self.epsilon:   #EPISLON GREEDY
            positions = list(range(64))
            np.random.shuffle(positions)
            for action in positions:
                pos = action // 8, action % 8
                if board.isValidMove(pid, pos[0], pos[1]):
                    valid_move = True
                    break
        else:
            sort_ind = torch.argsort(u, descending=True)

            for action in np.array(sort_ind[0]):
                pos = action // 8, action % 8
                if board.isValidMove(pid, pos[0], pos[1]):
                    valid_move = True
                    break


        if pos == False:
            return False
        if valid_move == False:
            return False
        elif log_history:
            # print("HISTORY____")
            # print(s)
            # print(len(np.where(s != 0)[0]))
            self.play_history.append((np.copy(input_nn), action, u[0][action]))
        return place_func(*pos)
        # try:
            # place_func(*pos)
            # return True
        # except TypeError:
            # return False

    def update_model(self, final_score):
        # state, action = self.play_history[0]
        optimizer = optim.Adam(self.nn.parameters())
        # q = self.nn.forward(state)
        # epochs = 2
        # for e in range(epochs):
        self.nn.train()
        pi_loss = []
        v_loss = []
        boards, pis, rs = list(zip(*self.play_history))
        target_pis = torch.FloatTensor(np.array(pis))
        target_rs = torch.FloatTensor(np.array(rs).astype(np.float64))

        pi_temp = []
        r_temp = []
        for i, s in enumerate(boards):

            u = self.nn.forward(torch.FloatTensor(s))
            u = u.reshape(1,u.shape[0])

            pi = torch.argmax(u)
            r = u[0][pi]
            pi_temp.append(pi)
            r_temp.append(r)

        pi_loss = self.loss_pi(target_pis, torch.FloatTensor(pi_temp))
        r_loss = self.loss_v(target_rs, torch.FloatTensor(r_temp))
        total_loss = pi_loss + r_loss
        # print(total_loss)
        # state_action_values = torch.FloatTensor(r_temp)
        # expected_state_action_values = (target_pis * self.gamma) + target_rs
        # loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)#.unsqueeze(1))
        loss = Variable(total_loss, requires_grad=True)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

class QLearningConvAgent:

    def __init__(self, alpha=0.07, gamma=0.99, k_max=50, explore_strategy='epsilon_greedy'):
        self.alpha = alpha
        self.gamma = gamma
        self.k_max = k_max
        self.explore_strategy = explore_strategy
        self.nn = conv.QLConvNet()
        self.wins = 0
        self.epsilon = 0.6
        self.play_history = []

    def play(self, place_func, board, pid, log_history=True):
        # populate the input array
        prev_board_state = board.getState()
        actions = []
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE):
                if board.isValidMove(pid, i, j):
                    temp = board.copy()
                    temp.updateBoard(pid, i, j)
                    board_state = temp.getState()
                    actions.append(board_state)
                else:
                    # may want to instead pass a zeroed out matrix
                    actions.append(prev_board_state)

        # (1, 64, 8, 8)
        conv_input = np.array(actions)
        # (1, 64)
        u = self.nn.forward(torch.unsqueeze(torch.FloatTensor(conv_input), 0))
        # choose the action that leads to the highest score
        pos = None
        valid_move = False
        if np.random.random() < self.epsilon:  # EPISLON GREEDY
            positions = list(range(64))
            np.random.shuffle(positions)
            for action in positions:
                pos = action // 8, action % 8
                if board.isValidMove(pid, pos[0], pos[1]):
                    valid_move = True
                    break
        else:
            sort_ind = torch.argsort(u, descending=True)
            for action in np.array(sort_ind[0]):
                pos = action // 8, action % 8
                if board.isValidMove(pid, pos[0], pos[1]):
                    valid_move = True
                    break

        if pos == False:
            return False
        if valid_move == False:
            return False
        elif log_history:
            raveled_pos = np.ravel_multi_index(pos, (8,8))
            self.play_history.append((np.copy(conv_input), raveled_pos, u[0][raveled_pos]))
        return place_func(*pos)


    def update_model(self, final_score):
        # state, action = self.play_history[0]
        optimizer = optim.Adam(self.nn.parameters())
        self.nn.train()
        pi_loss = []
        v_loss = []
        boards, pis, rs = list(zip(*self.play_history))
        target_pis = torch.FloatTensor(np.array(pis))
        target_rs = torch.FloatTensor(np.array(rs).astype(np.float64))

        pi_temp = []
        r_temp = []
        for i, s in enumerate(boards):
            u = self.nn.forward(torch.unsqueeze(torch.FloatTensor(s), 0))
            pi = torch.argmax(u)
            r = u[0][pi]
            pi_temp.append(pi)
            r_temp.append(r)
        pi_loss = self.loss_pi(target_pis, torch.FloatTensor(pi_temp))
        r_loss = self.loss_v(target_rs, torch.FloatTensor(r_temp))
        total_loss = pi_loss + r_loss
        # state_action_values = torch.FloatTensor(r_temp)
        # expected_state_action_values = (target_pis * self.gamma) + target_rs
        # loss = torch.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values)#.unsqueeze(1))
        loss = Variable(total_loss, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

                


# class ReplayGradientQLearningAgent:
# """
