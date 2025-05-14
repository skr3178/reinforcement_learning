from GameState import GameState


def  minimax(game_state : GameState, depth : int, maximizingPlayer : bool):

	if (depth==0) or (game_state.is_terminal()):
		return game_state.score(), None

	if maximizingPlayer:
		value = float('-inf')
		possible_moves = game_state.get_possible_moves()
		for move in possible_moves:
			child = game_state.get_new_state(move)

			tmp = minimax(child, depth-1, False)[0]
			if tmp > value:
				value = tmp
				best_movement = move

	else:
		value = float('inf')
		possible_moves = game_state.get_possible_moves()
		for move in possible_moves:
			child = game_state.get_new_state(move)

			tmp = minimax(child, depth-1, True)[0]
			if tmp < value:
				value = tmp
				best_movement = move

	return value, best_movement