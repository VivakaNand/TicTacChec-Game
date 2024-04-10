#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import random

# Define Tic Tac Chec constants
BOARD_SIZE = 4
NUM_PIECES = 8
WINNING_CONDITION = 4
NUM_ACTIONS = BOARD_SIZE**2 + NUM_PIECES

# Define Tic Tac Chec environment
class TicTacChecEnv:
    def __init__(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'White'
        self.moves_left = BOARD_SIZE**2

    def reset(self):
        self.board = [[' ' for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
        self.current_player = 'White'
        self.moves_left = BOARD_SIZE**2
        return self.get_state()

    def get_state(self):
        return np.array([[1 if cell == 'White' else -1 if cell == 'Black' else 0 for cell in row] for row in self.board])

    def is_valid_move(self, row, col, piece):
        return self.board[row][col] == ' ' and self.moves_left > 0

    def make_move(self, row, col, piece):
        if self.is_valid_move(row, col, piece):
            self.board[row][col] = piece
            self.moves_left -= 1
            return True
        return False

    def check_winner(self, player):
        # Check rows, columns, and diagonals for a win
        for i in range(BOARD_SIZE):
            for j in range(BOARD_SIZE - WINNING_CONDITION + 1):
                if all(self.board[i][j+k] == player for k in range(WINNING_CONDITION)) or                    all(self.board[j+k][i] == player for k in range(WINNING_CONDITION)):
                    return True

        for i in range(BOARD_SIZE - WINNING_CONDITION + 1):
            for j in range(BOARD_SIZE - WINNING_CONDITION + 1):
                if all(self.board[i+k][j+k] == player for k in range(WINNING_CONDITION)) or                    all(self.board[i+k][j+WINNING_CONDITION-1-k] == player for k in range(WINNING_CONDITION)):
                    return True

        return False

    def is_draw(self):
        return self.moves_left == 0

    def switch_player(self):
        self.current_player = 'Black' if self.current_player == 'White' else 'White'

# Define Q-learning agent
class QLearningAgent:
    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.2):
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.zeros((num_states, num_actions))
        self.state = None

    def choose_action(self, state):
        self.state = state
        if random.uniform(0, 1) < self.exploration_prob:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_table[state])

    def update_q_value(self, action, reward, next_state):
        max_next_q_value = np.max(self.q_table[next_state])
        self.q_table[self.state, action] += self.learning_rate * (
            reward + self.discount_factor * max_next_q_value - self.q_table[self.state, action]
        )

# Main training loop
def train_agent(agent, episodes):
    for episode in range(episodes):
        env = TicTacChecEnv()
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state = env.get_state()
            reward = 0
            if env.check_winner(1) or env.check_winner(-1):
                done = True
                if env.check_winner(1):
                    reward = 1
                else:
                    reward = -1
            elif env.is_draw():
                done = True

            agent.update_q_value(action, reward, next_state)
            state = next_state

# Initialize the Q-learning agent
num_states = BOARD_SIZE**2
num_actions = NUM_ACTIONS
agent = QLearningAgent(num_states, num_actions)

# Train the agent
train_agent(agent, episodes=10000)

# We can now use the trained agent to play the game or evaluate its performance.


# In[ ]:


def play_game(agent, env):
    state = env.reset()
    done = False

    while not done:
        if env.current_player == 'White':
            # Agent's turn
            action = agent.choose_action(state)
            row, col, piece = action // BOARD_SIZE, action % BOARD_SIZE, 'White'
            print(f"Agent's move: {row}, {col}")
        else:
            # Human's turn (you can replace this with user input)
            print("Human's turn (row, col): ")
            row, col = map(int, input().split())
            piece = 'Black'

        if env.is_valid_move(row, col, piece):
            env.make_move(row, col, piece)

        env.switch_player()
        state = env.get_state()

        if env.check_winner(1):
            print("Agent wins!")
            done = True
        elif env.check_winner(-1):
            print("Human wins!")
            done = True
        elif env.is_draw():
            print("It's a draw!")
            done = True

# Initialize the environment and agent
env = TicTacChecEnv()
agent = QLearningAgent(num_states, num_actions)

# Load the trained Q-table (if available)
# agent.q_table = np.load("q_table.npy")

# Play a game
play_game(agent, env)

# Save the Q-table (optional)
# np.save("q_table.npy", agent.q_table)

