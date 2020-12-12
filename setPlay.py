from model import *
import MCTS
import env
import chess
import chess.pgn
import chess.uci
import os
from tqdm import tqdm
from time import time

agent = load_model("long_ResNet64.json", "long_ResNet64.h5")
#Init Stockfish
stockfish = chess.uci.popen_engine("stockfish")
stockfish.uci()

games = 25 #Number of games to play
simulations = 1600 #Node count per move



white = True
start = time()
for it in tqdm(range(games)):
	board = chess.Board()
	turn = white
	while not board.is_game_over():
		if turn:
			stockfish.position(board)
			stock_move = stockfish.go(movetime=100) #Set stockfish to 10 moves per second
			board.push(stock_move[0])
		else:
			policy, values, _, _ = MCTS.muMCTS(board, board.turn, agent, simulations=simulations)
			index = np.random.choice(1968, p=policy)
			uci_move = env.MOVES[index]
			board.push_uci(uci_move)

		turn = not turn

	#Save a pgn of the game for analysis later
	game = chess.pgn.Game().from_board(board)
	game.headers["Event"] = "Game " + str(it+1)
	game.headers["Site"] = "RTX Titan & AMD ThreadRipper"
	game.headers["Round"] = "Testing (1600 Nodes vs 100ms)"

	if white:
		game.headers["White"] = "Stockfish 12"
		game.headers["Black"] = "Basilisk"
	else:
		game.headers["White"] = "Basilisk"
		game.headers["Black"] = "Stockfish 12"

	name = "games/testing/1600N_c=10.0/" + str(it+1) + ".pgn"
	directory = "games/testing/1600N_c=10.0/"
	
	try:
		os.mkdir(directory)
	except FileExistsError:
		#directory already exists
		pass

	new_pgn = open(name, "w", encoding="utf-8")
	exporter = chess.pgn.FileExporter(new_pgn)
	game.accept(exporter)

	white = not white #Alternate who plays white

print("Time (Minutes)", (time() - start)/60)