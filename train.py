import numpy as np
from tensorflow import random as rnd
import random
from customLib import ConnectFourGame, DQN
from tensorflow.keras.optimizers import Adam
seed = 1
np.random.seed(seed)
rnd.set_seed(seed)
random.seed(seed)
'''Training loop'''
episodes = 10000  # Can play around with this and forget_buffer to see what changes
size = [6, 7]
batch_size = 200
forget_buffer = 5  # For how
playerOne = DQN(ddqn=True, size=size, episodes=episodes-10)  # Roughly amount of time before enough data is acquired to learn
playerTwo = DQN(ddqn=True, size=size, episodes=episodes-10)
playerOne.weights_file = "../../Downloads/P1.h5"
playerTwo.weights_file = "../../Downloads/P2.h5"
board = ConnectFourGame(size)
board.display_board()
print(board.simplify())
movesOneList = []
movesTwoList = []
for episode in range(episodes):
    print("Episode " + str(episode + 1))
    # Multi-agent play
    movesOne = 0
    movesTwo = 0
    while (not board.board.winCheck("X")) or (not board.board.winCheck("O")):
            state = (board.simplify()).reshape((1, size[0], size[1], 3))
            actionOne = playerOne.act(state)
            rewardOne = board.drop_piece("X", actionOne)
            next_state = (board.simplify()).reshape((1, size[0], size[1], 3))
            playerOne.remember(state, actionOne, rewardOne, next_state, (board.board.winCheck("X") or board.board.winCheck("O")))
            movesOne += 1
            if board.board.winCheck("X"):  # Winning move
                playerTwo.forget_recent(1)
                playerTwo.remember(next_state, actionTwo, -10, next_next_state, (board.board.winCheck("X") or board.board.winCheck("O")))
                break
            elif movesOne + movesTwo == size[0]*size[1]:  # Board completely filled up
                break
            actionTwo = playerTwo.act(next_state)
            rewardTwo = board.drop_piece("O", actionTwo)
            next_next_state = (board.simplify()).reshape((1, size[0], size[1], 3))
            playerTwo.remember(next_state, actionTwo, rewardTwo, next_next_state, (board.board.winCheck("X") or board.board.winCheck("O")))
            movesTwo += 1
            if board.board.winCheck("O"):  # Winning move
                playerOne.forget_recent(1)
                playerOne.remember(state, actionOne, -10, next_state, (board.board.winCheck("X") or board.board.winCheck("O")))
                break
            elif movesOne + movesTwo == size[0]*size[1]:
                break
    board.display_board()  # What the board looked like at the end of the game
    movesOneList.append(movesOne)
    movesTwoList.append(movesTwo)
    print("Moves: " + str(movesOne + movesTwo))
    if len(playerOne.memory) > forget_buffer*batch_size:  # Discards data that was acquired when opponent was less trained. Also easier on computer.
            playerOne.forget(movesOneList[0])
            movesOneList.pop(0)
    if len(playerTwo.memory) > forget_buffer*batch_size:  # Same for player two
            playerTwo.forget(movesTwoList[0])
            movesTwoList.pop(0)
    if len(playerOne.memory) > batch_size:
            playerOne.replay(batch_size)
    if len(playerTwo.memory) > batch_size:
            playerTwo.replay(batch_size)
    board.board.cleanup()  # New board and formatting
    print(" ")
playerOne.save_weights()
playerTwo.save_weights()
