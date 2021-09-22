# Connect4RL
Repository for Reinforcement Learning Agents I made for playing Connect 4 by themselves.<br>
This is a multi-agent reinforcement learning problem, as Connect 4 is a two player game where the moves that you make are very much based on the moves your opponent will make. As such, a forget buffer is implemented (not just for computational ease, although that is a big factor).<br>
The customLib.py file contains the Connect 4 game engine that I made a long time ago (April 2020, soon after start of quarantine) and the training loop I added on to make the two Connect Four Agents learn by playing against each other.<br>
Both agents use DDQN architecture with a Convolutional Representation of the board as an input and the column to drop the piece in as output.

# The h5 Files in This Repository
All the .h5 files in this repository are weight files, and their names will detail which player (X or O) they are playing for (P1 for X and P2 for O), how many episodes they trained for, and the value of forget buffer times batch_size, respectivley. You can test each model with test.py
