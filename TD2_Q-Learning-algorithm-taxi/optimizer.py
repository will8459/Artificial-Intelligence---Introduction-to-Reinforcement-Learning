import numpy as np
import itertools
from q_learning import train, evaluate 

"""
This script performs a Grid Search to find the optimal hyperparameters
(alpha, gamma, epsilon) for the Q-Learning agent defined in 'q_learning.py'.

Usage:
1. Run this script.
2. Wait for the search to complete.
3. Update the variables in 'q_learning.py' with the "Best Parameters" found.
"""

def grid_search():
    """
    Performs a Grid Search to find the best hyper-parameters.
    """
    # 1. Define the search space
    param_grid = {
        'alpha': [0.2, 0.3, 0.4],
        'gamma': [0.92, 0.95, 0.97],
        'epsilon': [0.4, 0.5, 0.6]
    }

    # Generate all combinations
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Testing {len(combinations)} combinations...")
    
    best_score = -np.inf
    best_params = {}

    for i, params in enumerate(combinations):
        print(f"Test {i}/{len(combinations)} : {params} | ", end="")
        
        # 1. Train with specific parameters
        Q, _ = train(params['alpha'], params['gamma'], params['epsilon'], n_epochs=20000)
        
        # 2. Evaluate (no render)
        # Average over 100 episodes for statistical stability
        score, _ = evaluate(Q, n_episodes=100, render=False)
        
        print(f" Score: {score}")

        # 3. Track the best
        if score > best_score:
            best_score = score
            best_params = params

    print("\nRESULTS :")
    print(f"Best Score: {best_score}")
    print(f"Best Parameters: {best_params}\n")
    print("Please update these values manually in q_learning.py")

if __name__ == "__main__":
    grid_search()