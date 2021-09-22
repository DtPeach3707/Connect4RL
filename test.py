'''
Code for testing the agent
'''
from customLib import ConnectFourGame, DQN
size = [6, 7]  # Make sure the size is the same as the size that you trained the model on
agent = DQN(ddqn=True, size=size)
agent.epsilon = 0.2  # Reducing probability of random action
board = ConnectFourGame(size=size)
board.display_board()
agent_p1 = True
weights_fileOne = "../../Downloads/P1.h5"  # Path to each player file. Will need to change either file name or these lines when implementing for yourself.
weights_fileTwo = "../../Downloads/P2.h5"
#Agent play
movesOne = 0
movesTwo = 0
valid = False
if agent_p1:
    agent.q_model.load_weights(weights_fileOne)
    while (not board.board.winCheck('X')) or (not board.board.winCheck('O')):
        next_state = (board.simplify()).reshape((1, 6, 7, 3))
        actionTwo = agent.act(next_state)
        rewardTwo = board.drop_piece("X", actionTwo)
        board.display_board()
        next_next_state = (board.simplify()).reshape((1, 6, 7, 3))
        movesTwo += 1
        while not valid:  # So wrong input doesn't terminate the code
            inputs= input("Pick a row(1-" + str(size[0]) + "):")
            try:
                inputs = int(inputs)
                if not (inputs <= 0 or inputs > size[0]):
                    valid = True
                else:
                    print("Make sure to choose within the limits of the board")
            except ValueError:
                print("Enter a valid integer please")
        board.drop_piece("O", inputs)
        board.display_board()
else:
    agent.q_model.load_weights(weights_fileTwo)
    while (not board.board.winCheck('X')) or (not board.board.winCheck('O')):
        next_state = (board.simplify()).reshape((1, 6, 7, 3))
        while not valid:  # So wrong input doesn't terminate the code
            inputs= input("Pick a row(1-" + str(size[0]) + "):")
            try:
                inputs = int(inputs)
                if not (inputs <= 0 or inputs > size[0]):
                    valid = True
                else:
                    print("Make sure to choose within the limits of the board")
            except ValueError:
                print("Enter a valid integer please")
        movesTwo += 1
        board.drop_piece("O", int(inputs))
        board.display_board()
        next_next_state = (board.simplify()).reshape((1, 6, 7, 3))
        actionTwo = agent.act(next_state)
        rewardTwo = board.drop_piece("X", actionTwo)
        board.display_board()
