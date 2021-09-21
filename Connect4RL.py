from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D
from tensorflow.keras.models import Model
import numpy as np
from tensorflow import random as rnd
import random
from tensorflow.keras.optimizers import Adam
seed = 1
np.random.seed(seed)
rnd.set_seed(seed)
random.seed(seed)


class ConnectFourBoard:
    def __init__(self, size=[7, 6]):  # Size set to default Connect Four Board Size
        self.size = size
        if not (self.size[0] >= 6 and self.size[1] >= 6):  # Minimum size is 6x6 for this
            if self.size[0] < 6:
                print("Minimum row size is 6, so the size is increased to 6")
                self.size[0] = 6
            if self.size[1] < 6:
                print("Minimum column size is 6, so the size is increased to 6")
                self.size[1] = 6
        self.size = tuple(self.size)
        self.board = []
        for n in range(
                self.size[1]):  # Setting up the board for display and iterating for dropping pieces and win checks
            self.board.append([])
        for row in self.board:
            for n in range(self.size[0]):
                row.append('_')  # Indicates that the given slot is blank

    def display_board(self):
        num_lis = [num for num in range(self.size[1])][::-1]  # Flipped to make drop_piece easier to code
        for num in num_lis:
            print(self.board[num])

    def drop_piece(self, name, col):
        placed = False
        for row in self.board:  # Iterates from bottom to top of the board
            if row[col - 1] == '_':  # Looks for the first blank space in the column
                row[col - 1] = name
                placed = True
                break  # Only drops one piece in one column
        if not placed:  # Picks closest adjacent column
            rangn = []  # How it goes about picking a new column when the one it picked is filled up
            for i in range(self.size[0] - 1):
                if i % 2 == 0:  # If divisible by 2 (includes 0)
                    rangn.append(int(i/2))
                else:  # If an odd number
                    rangn.append(-2 - (int((i-1)/2)))
            filled = False
            for ran in rangn:
                for row in self.board:
                    if row[((col + ran) % self.size[0])] == '_':
                        row[((col + ran) % self.size[0])] = name
                        filled = True
                        break
                if filled:
                    break
        if self.winCheck(name):  # Reward integrated in drop_piece
            return 10
        else:
            return 1

    def winCheck(self, name):
        for row in self.board:  # Horizontal win
            for col in range(self.size[0] - 3):
                if row[col] == name and row[col + 1] == name and row[col + 2] == name and row[col + 3] == name:
                    return True
        for row in range(self.size[1] - 3):  # Vertical win
            for col in range(self.size[0]):
                if self.board[row][col] == name and self.board[row + 1][col] == name and self.board[row + 2][col] == name and self.board[row + 3][col] == name:
                    return True
        for row in range(self.size[1] - 3):  # Diagonal wins
            for col in range(self.size[0] - 3):
                if self.board[row][col] == name and self.board[row + 1][col + 1] == name and self.board[row + 2][col + 2] == name and self.board[row + 3][col + 3] == name:
                    return True
                if self.board[row][col + 3] == name and self.board[row + 1][col + 2] == name and self.board[row + 2][col + 1] == name and self.board[row + 3][col] == name:
                    return True
        return False  # If it isn't true . . .

    def simplify(self):
        simp_board = []
        for row in self.board:
            simp_row = []
            for space in row:
                if space == 'O':
                    simp_row.append([0, 0, 1])
                elif space == 'X':
                    simp_row.append([0, 1, 0])
                else:
                    simp_row.append([1, 0, 0])
            simp_board.append(simp_row)
        return np.array((simp_board))

    def cleanup(self):  # Remakes board
        self.board = []
        for n in range(self.size[1]):  # Setting up the board for display and iterating for dropping pieces and win checks
            self.board.append([])
        for row in self.board:
            for n in range(self.size[0]):
                row.append('_')  # Indicates that the given slot is blank


class ConnectFourGame:
    def __init__(self, size=[7, 6]):
        self.board = ConnectFourBoard(size=size)
        self.stop = False

    def display_board(self):
        print(self.board.display_board())

    def drop_piece(self, name, col):
        if self.stop == False:
            return self.board.drop_piece(name, col)
        else:
            print("The game has ended. You cannot place anymore pieces")

    def reward(self, name, opponent):
        if self.stop == False:
            if self.board.winCheck(name):
                return 10
            elif self.board.winCheck(opponent):
                return -10
            else:
                return -1
        else:
            print("The game has ended. You can no longer get rewarded.")

    def simplify(self):
        return self.board.simplify()


class DQN():
    def __init__(self, ddqn, drift=False, episodes=2000, size=[7, 6]):
        self.size = size
        if not (self.size[0] >= 6 and self.size[1] >= 6):  # Minimum size is 6x6 for this
            if self.size[0] < 6:
                print("Minimum row size is 6, so the size is increased to 6")
                self.size[0] = 6
        if self.size[1] < 6:
            print("Minimum column size is 6, so the size is increased to 6")
            self.size[1] = 6
        self.size = tuple(self.size)
        # experience buffer
        self.memory = []

        # discount rate
        self.gamma = 0.9
        self.size = size
        # initially 90% exploration, 10% exploitation
        self.epsilon = 0.9
        # iteratively applying decay til 10% exploration/90% exploitation
        self.epsilon_min = 0.1
        self.epsilon_decay = self.epsilon_min / self.epsilon
        self.epsilon_decay = self.epsilon_decay ** (1. / float(episodes))

        # Q Network weights filename
        self.weights_file = 'ddqn_MKDS.h5' if ddqn else 'dqn_MKDS.h5'
        self.n_outputs = self.size[0]
        # Q Network for training
        self.q_model = self.build_model(self.n_outputs)
        self.q_model.compile(loss='mse', optimizer=Adam())
        # target Q Network
        self.target_q_model = self.build_model(self.n_outputs)
        # copy Q Network params to target Q Network
        self.update_weights()

        self.replay_counter = 0
        self.ddqn = True if ddqn else False
        if self.ddqn:  # Loads in weights file if there is one
            print("----------Double DQN--------")
        else:
            print("-------------DQN------------")

    def build_model(self, n_outputs):  # Network architecture
        inputs = Input(shape=(self.size[0], self.size[1], 3), name="state")
        conv = Conv2D(64, kernel_size=5, padding="same", activation='relu')(inputs)
        conv = Conv2D(64, kernel_size=5, padding="same", activation='relu')(conv)
        conv = Conv2D(32, kernel_size=3, padding="same", activation='relu')(conv)
        x = Flatten()(conv)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(n_outputs, activation='linear', name='action')(x)
        q_model = Model(inputs, x)
        q_model.summary()
        return q_model

    # save Q Network params to a file
    def save_weights(self):
        self.q_model.save_weights(self.weights_file)

    def update_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())

    # eps-greedy policy
    def act(self, state):
        if np.random.rand() < self.epsilon:
            rand_action = np.random.choice(self.size[0])
            return rand_action

        # exploit
        q_values = self.target_q_model.predict(state)
        best_action = np.argmax(q_values[0])
        return best_action

    # store experiences in the replay buffer
    def remember(self, state, action, reward, next_state, done):
        item = (state, action, reward, next_state, done)
        self.memory.append(item)

    # compute Q_max
    # use of target Q Network solves the non-stationarity problem

    def forget(self, length):
        for i in range(length):
            self.memory.pop(0)

    def forget_recent(self, length):
        for i in range(length):
            self.memory.pop(-1)

    def get_target_q_value(self, next_state, reward):
        # max Q value among next state's actions
        if self.ddqn:
            # current Q Network selects the action
            # a'_max = argmax_a' Q(s', a')
            action = np.argmax(self.q_model.predict(next_state)[0])
            # target Q Network evaluates the action
            # Q_max = Q_target(s', a'_max)
            q_value = self.target_q_model.predict(next_state)[0][action]
        else:
            q_value = np.amax(self.target_q_model.predict(next_state)[0])

        # Q_max = reward + gamma * Q_max
        q_value *= self.gamma
        q_value += reward
        return q_value

    # experience replay addresses the correlation issue between samples
    def replay(self, batch_size):
        # sars = state, action, reward, state' (next_state)
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)

            # get Q_max
            q_value = self.get_target_q_value(next_state, reward)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value
            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch),
                         np.array(q_values_batch),
                         batch_size=batch_size,
                         epochs=1,
                         verbose=True)
        # update exploration-exploitation probability
        self.update_epsilon()
        # copy new params on old target after every 3 training updates
        if self.replay_counter % 5 == 0:
            self.update_weights()

        self.replay_counter += 1

    # decrease the exploration, increase exploitation

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


'''Training loop'''
episodes = 10000  # Can play around with this and forget_buffer to see what changes
size = [6, 7]
batch_size = 200
forget_buffer = 5  # For how
playerOne = DQN(ddqn=True, size=size, episodes=episodes-10)
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