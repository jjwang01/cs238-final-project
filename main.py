import numpy as np
import sys

from game import Game
from player import HumanPlayer, RandomAgent, PositionalAgent, MobilityAgent

# use argparse down the line
if len(sys.argv) != 5:
    print("Usage should be main.py <player_agent> <opp_agent> <num_matches>")
    sys.exit()

if sys.argv[1] == 'random':
    player = RandomAgent()
else:
    print("No such agent.")
    sys.exit()

if sys.argv[2] == 'random':
    opp = RandomAgent()
else:
    print("No such agent.")
    sys.exit()

try:
    num_matches = int(sys.argv[3])
except ValueError:
    print("num_matches must be an integer")
    sys.exit()


player_game_history = []
for _ in range(num_matches):
    player.play_history = []

    G = Game()
    G.addPlayer(player)
    G.addPlayer(opp)
    G.run(show_board=True)

    final_score = list(G.getScore().items())
    final_score.sort()
    total = sum(map(lambda x: x[1], final_score))
    player_score = (final_score[0][1]/total - 0.5) * 2
    player.wins += player_score > 0
    player_game_history.append((player.play_history, player_score))

print('The player wins ' + str(player.wins) + ' out of ' + str(num_matches) + ' times.')

