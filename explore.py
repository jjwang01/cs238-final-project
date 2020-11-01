def epsilon_greedy(place_func):
    # epsilon greedy to pick random move
    if np.random.random() < self.epsilon:
        positions = list(itertools.product(range(8), repeat=2))
        random.shuffle(positions)
        while not made_move and positions:
            pos = positions.pop()
            made_move = place_func(*pos)

        # If we can make no move... pass
        if not made_move and not positions:
            return False
    else:
        out = self.policy_net.getOutput(input_state)
        # Sort the possible moves lowest to highest desire
        positions = [(v, i) for i, v in enumerate(out)]
        positions.sort(key=lambda x: x[0], reverse=True)

        while not made_move and positions:
            # Grab next desired move point
            scalar_play_point = positions.pop()[1]
            # Convert the scalar to a 2D coordinate to play on the board
            pos = scalar_play_point // 8, scalar_play_point % 8
            made_move = place_func(*pos)

        # If we can make no move... pass
        if not made_move and not positions:
            return False
        
    return pos