# machine-learning-project


Taxi-v3 Q-Learning with Pygame GUI
This project implements a reinforcement learning agent using the Q-learning algorithm to solve the Taxi-v3 environment from OpenAI Gym. The agent learns to pick up and drop off passengers in a 5x5 grid world.

A custom graphical interface built with Pygame visually displays the taxi's movements, pickup, and drop-off actions in real-time.

Features
Q-learning with epsilon-greedy policy and decay

Trains a Q-table or loads a pre-trained one (q_table.npy)

Visual representation of the grid, taxi, pickup, and drop-off locations

Step-by-step movement and interaction feedback in the console

How it Works
The Q-table is trained for 25,000 episodes if not already saved.

The trained agent then runs one complete episode in the GUI.

The GUI updates in real-time, showing the taxi navigating, picking up the passenger, and dropping them off.

Requirements
Python 3.x

gym

numpy

pygame

To run the project, simply execute the script. If no Q-table exists, it will be trained automatically.
