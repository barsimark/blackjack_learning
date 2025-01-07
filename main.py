import numpy as np
import gym

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Initialize Q-table
state_space = [32, 11, 2] # Hand total, dealer upcard, hard or soft hand (without or with ace; 0 or 1)
action_space = 2 # 0:stand, 1:hit #TODO extend action space with double, split, surround
q_table = np.zeros(state_space + [action_space])

optimal_basic_strategy = np.zeros(state_space, dtype=int)

# Setup optimal basic strategy for stand/hit only blackjack
for player_total in range(2, 33):
    for dealer_upcard in range(1, 11):
        for hand_type in range(2):
            if hand_type == 0:  # Hard hand
                if player_total >= 17:
                    optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type] = 0  # Stand
                elif 13 <= player_total <= 16 and dealer_upcard <= 6:
                    optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type] = 0  # Stand
                elif 12 == player_total and 4 <= dealer_upcard <= 6:
                    optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type] = 0  # Stand
                else:
                    optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type] = 1  # Hit
            else:  # Soft hand
                if player_total >= 19:
                    optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type] = 0  # Stand
                elif player_total == 18 and 3 <= dealer_upcard <= 6:
                    optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type] = 0  # Stand
                else:
                    optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type] = 1  # Hit

# Create the environment
env = gym.make('Blackjack-v1')
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.2  # Exploration-exploitation tradeoff
episodes = 1

# Training loop
for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # If state is the first in the episode, it contains an extra empty dictionary
        if len(state) == 2:
            player_sum, dealer_card, usable_ace = state[0]
        else:
            player_sum, dealer_card, usable_ace = state
            
        # Choose action according to epsilon
        if np.random.rand() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[player_sum, dealer_card, int(usable_ace)])  # Exploit
        
        # Get next state in the episode
        next_state, reward, done, _, _ = env.step(action)
        next_player_sum, next_dealer_card, next_usable_ace = next_state
        
        # Update Q-table
        q_table[player_sum, dealer_card, int(usable_ace), action] += alpha * (
            reward + gamma * np.max(q_table[next_player_sum, next_dealer_card, int(next_usable_ace)]) - 
            q_table[player_sum, dealer_card, int(usable_ace), action]
        )
        
        # Keep going to next state of the episode
        state = next_state
        print(state, action, next_state, reward)
    
np.save('q_table.npy', q_table)
    
num_differences = 0

# Compare the optimal strategy with the Q-table
for player_total in range(2, 32):  # Player total ranges from 2 to 32
    for dealer_upcard in range(1, 11):  # Dealer upcard ranges from 1 to 10
        for hand_type in range(2):  # 0 = Hard, 1 = Soft
            # Get the optimal action (0 or 1)
            optimal_action = optimal_basic_strategy[player_total - 2, dealer_upcard, hand_type]
            
            # Get the action with the highest Q-value for the given state
            best_q_action = np.argmax(q_table[player_total - 2, dealer_upcard, hand_type])
            
            # Compare and count differences
            if optimal_action != best_q_action:
                num_differences += 1

print(f"Number of differences between Q-table and optimal strategy: {num_differences}")