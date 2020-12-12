#This is our multiprocessing loop!
def play_from(board, pos, moves, res, lengths, lock):
	import chess
	import chess.uci
	import numpy as np 
	from env import convert, generate_move_dict, fast_onehot
	import sparse 


	stockfish = chess.uci.popen_engine("stockfish")
	stockfish.uci()

	s_pos = []
	s_moves = []
	result = 0.0 #Value is not relative
	move_dict = generate_move_dict()
	length = 0.0 #Record the length of the trajectory

	while not (board.is_game_over() or board.can_claim_draw()):
		stockfish.position(board)
		stock_move = stockfish.go(movetime=100) #Set stockfish to 10 moves per second

		position = convert(board) #convert to position
		s_pos.append(position) #Append position to our temporary list

		new_move = board.uci(stock_move[0])
		s_moves.append(move_dict[new_move]) #Append stockfish label as onehot vector

		board.push(stock_move[0])
		length += 1

	if board.is_checkmate():
		if board.turn:
			result = -1.0
		else:
			result = 1.0
	else:
		result = 0.0

	#use lock to add new data to master list 
	lock.acquire()
	index = 0
	lengths.append(length) #Use this data for analysis of expert trajectories later
	end = len(s_pos)
	for p in s_pos:
		coeff = np.power(0.99, end - index - 1) #Create labels using Deterministic Bellman Function
		res.append(result*coeff)
		pos.append(p)
		moves.append(s_moves[index])
		index += 1
	lock.release()

if __name__ == "__main__":
	#Import statements
	import chess
	import chess.uci
	import chess.pgn

	from tqdm import tqdm
	from time import time
	
	from multiprocessing import Process, Manager, Lock

	import numpy as np
	import os
	import env

	from scipy import sparse 

	board = chess.Board()
	manager = Manager()
	lock = Lock()
	num_processes = 5000
	num_workers = 32
	move_dict = env.generate_move_dict()

	model_image = manager.list()
	expert = manager.list()
	targ = manager.list()
	lengths = manager.list()

	#Init Stockfish
	stockfish = chess.uci.popen_engine("stockfish")
	stockfish.uci()
	info_handler = chess.uci.InfoHandler()
	stockfish.info_handlers.append(info_handler)

	base_length = 20 #Length of base trajectories
	processes = []

	#Create a dictionary
	fen_dict = {} #This will make sure we don't create a process on a position we have already observed

	#Begin Simulation of Games
	print("Siumulating Games...")
	t0 = time()
	for m in tqdm(range(num_processes)): 
		board.reset()
		ply_number = 1

		while not (board.is_game_over() or ply_number >= base_length+1):
			copy = board.copy() #Create a copy of the board to be played in parallel
			fen = copy.fen()
			if not fen in fen_dict: #If the position is unique
				p = Process(target=play_from, args=(copy, model_image, expert, targ, lengths, lock)) #Generate labels in parallel
				p.start()
				processes.append(p)
				base_policy = np.zeros(1968)
				bitmask = np.zeros(1968)
				for move in board.legal_moves:
					board.push(move) #Push move to get a child position for evaluation

					key = board.uci(move) #Get uci key for dictionary
					index = move_dict[key] #Get index of policy value
					bitmask[index] = 1 #Fill bitmask
					stockfish.position(board)
					_ = stockfish.go(movetime=5) #Evaluate each move at 5ms to create a policy
					score = info_handler.info["score"][1]
					if score.mate is None: #If its not checkmate
						cp = float(score.cp)
						base_policy[index] = -cp/10
					else: #If stockfish detects a mate prioritize it
						val = np.sign(score.mate)*100
						base_policy[index] = -val
					board.pop() #Undo child move

				base_policy = np.exp(base_policy - np.max(base_policy)) #Numerically Softmax to create a discrete distribution
				base_policy = base_policy * bitmask #bitmask is applied after to ensure only legal moves are selected
				base_policy = base_policy/np.sum(base_policy) #Normalize

				index = np.random.choice(1968, p=base_policy) #Sample from the base policy
				fen_dict[fen] = base_policy
			else:
				base_policy = fen_dict[fen]
				index = np.random.choice(1968, p=base_policy) #Sample from the base policy

			board.push_uci(env.MOVES[index]) #Push the selected move to the board environment
			while len(processes) > num_workers: #Don't let there be more than 31 processes being used
				for idx in range(len(processes)): #Close dead processes
					if not processes[idx].is_alive():
						processes[idx].join()
						processes.pop(idx)
						break
			

			ply_number += 1

		game = chess.pgn.Game().from_board(board)
		game.headers["Event"] = "Base Trajectory " + str(m+1)
		game.headers["Site"] = "RTX Titan"
		game.headers["Round"] = "Training"
		game.headers["White"] = "base_policy"
		game.headers["Black"] = "base_policy"

		name = "games/BaseTraj" + str(m+1) + ".pgn"
		directory = "games/"
		
		try:
			os.mkdir(directory)
		except FileExistsError:
			#directory already exists
			pass

		new_pgn = open(name, "w", encoding="utf-8")
		exporter = chess.pgn.FileExporter(new_pgn)
		game.accept(exporter)


	print("Cleaning up...")
	for i in range(len(processes)):
		processes[i].join()

	processes = [] #clear memory
	#Load data
	print("Loading Data...")
	model_image = np.reshape(np.array(model_image), [-1, 8, 8, 17])
	expert = np.reshape(np.array(expert), [-1, 1])
	targ = np.reshape(np.array(targ), [-1, 1])
	lengths = np.reshape(np.array(lengths), [-1, 1])

	print("Saving Data")
	np.save('sparse_model_input', model_image)
	np.save('sparse_expert_policy', expert)
	np.save('sparse_state_value', targ)
	np.save('sparse_lengths', lengths)

	t1 = time()
	print("Time: ", t1 - t0) #Check how much time it takes to generate a data set.