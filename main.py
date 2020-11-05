import numpy as np
import sys

from game import Game
from player import HumanPlayer, RandomAgent, PositionalAgent, MobilityAgent, QLearningAgent

# use argparse down the line
if len(sys.argv) != 4:
    print("Usage should be main.py <player_agent> <opp_agent> <num_matches>")
    sys.exit()

qlearn_pl = False 
qlearn_opp = False

if sys.argv[1] == 'random':
    player = RandomAgent()
elif sys.argv[1] == 'positional':
    player = PositionalAgent()
elif sys.argv[1] == 'mobility':
    player = MobilityAgent()
elif sys.argv[1] == 'qlearn':
    player = QLearningAgent()
    qlearn_pl = True
else:
    print("No such agent.")
    sys.exit()

if sys.argv[2] == 'random':
    opp = RandomAgent()
elif sys.argv[2] == 'positional':
    opp = PositionalAgent()
elif sys.argv[2] == 'mobility':
    opp = MobilityAgent()
elif sys.argv[2] == 'qlearn':
    opp = QLearningAgent()
    qlearn_opp = True
else:
    print("No such agent.")
    sys.exit()

try:
    num_matches = int(sys.argv[3])
except ValueError:
    print("num_matches must be an integer")
    sys.exit()

num_epochs = 1

if qlearn_pl or qlearn_opp:
    num_epochs = 20

player_wins = []

for e in range(num_epochs):
    player.wins = 0
    player_game_history = []

    for _ in range(num_matches):
        player.play_history = []

        G = Game()
        G.addPlayer(player)
        G.addPlayer(opp)
        # G.run(show_board=True)
        G.run()

        final_score = list(G.getScore().items())
        final_score.sort()
        total = sum(map(lambda x: x[1], final_score))
        player_score = (final_score[0][1]/total - 0.5) * 2
        player.wins += player_score > 0
        player_game_history.append((player.play_history, player_score))

    for game, score in player_game_history:
        player.play_history = game
        if qlearn_pl:
            player.update_model(score)
        if qlearn_opp:
            opp.update_model(score)

    print('The player wins ' + str(player.wins) + ' out of ' + str(num_matches) + ' times.')

