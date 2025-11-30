import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


def update_q_table(Q, s, a, r, sprime, alpha, gamma):
    """
    This function should update the Q function for a given pair of action-state
    following the q-learning algorithm, it takes as input the Q function, the pair action-state,
    the reward, the next state sprime, alpha the learning rate and gamma the discount factor.
    Return the same input Q but updated for the pair s and a.
    """

    # Calculate the max Q-value for the next state (sprime)
    max_future_q = np.max(Q[sprime, :])
    
    # Update Q-value using the Bellman equation
    Q[s, a] = Q[s, a] + alpha * (r + gamma * max_future_q - Q[s, a])
    
    return Q


def epsilon_greedy(Q, s, epsilone):
    """
    This function implements the epsilon greedy algorithm.
    Takes as unput the Q function for all states, a state s, and epsilon.
    It should return the action to take following the epsilon greedy algorithm.
    """
    
    # Explore: choose a random action with probability epsilone
    if np.random.rand() < epsilone:

        action = np.random.randint(0, Q.shape[1])
    else:
        # Exploit: choose the best known action with probability 1-epsilon
        action = np.argmax(Q[s, :])
        
    return action

# separate functions in order to be able to import them in other files

def train(alpha, gamma, epsilon, n_epochs=10000, max_itr_per_epoch=100):
    """
    Encapsulated training function. 
    Runs the Q-Learning training loop and returns the trained Q-table and reward history.
    """
    # Create environment (no render_mode to speed up training)
    env = gym.make("Taxi-v3")
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    rewards = []

    for e in range(n_epochs):
        r = 0
        S, _ = env.reset()

        for _ in range(max_itr_per_epoch):
            A = epsilon_greedy(Q=Q, s=S, epsilone=epsilon)
            Sprime, R, done, _, _ = env.step(A)
            
            r += R
            
            Q = update_q_table(Q, S, A, R, Sprime, alpha, gamma)
            
            # Update state
            S = Sprime
            
            if done:
                break
        
        # print reward per episode
        # print("train_episode #", e, " : r = ", r) 
        
        rewards.append(r)
        
    env.close()
    return Q, rewards

def evaluate(Q, n_episodes=10, max_itr=100, render=False):
    """
    Evaluation function using a purely Greedy policy.
    Returns the mean reward over n_episodes.
    """
    render_mode = "human" if render else None
    env = gym.make("Taxi-v3", render_mode=render_mode)
    eval_rewards = []

    for e in range(n_episodes):
        S, _ = env.reset()
        r = 0
        done = False
        
        for _ in range(max_itr):
            # Greedy policy (exploitation only)
            A = np.argmax(Q[S, :]) 
            
            Sprime, R, done, _, _ = env.step(A)
            r += R
            S = Sprime
            
            if render:
                env.render()
                
            if done:
                break

        if render:
            # print reward per episode
            print("eval_episode #", e, " : r = ", r)

        eval_rewards.append(r)
    
    env.close()
    return np.mean(eval_rewards), eval_rewards

if __name__ == "__main__":    
    """
    Default hyper-parameters.
    Update these after running 'optimizer.py'.
    """
    alpha = 0.3      # Learning rate
    gamma = 0.95     # Discount factor
    epsilon = 0.5    # Exploration rate
    
    print(f"Start training with Learning rate={alpha}, Discount factor={gamma}, Exploration rate={epsilon} :")

    # 1. Train the agent
    Q_learned, training_rewards = train(alpha, gamma, epsilon, n_epochs=20000)
    
    print(f"Average reward: {np.mean(training_rewards)}\nTraining finished.")
    
    # Plot the rewards in function of epochs
    print("Plotting training rewards... \n")
    plt.figure()
    plt.plot(training_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Training Progress")
    plt.show()

    # 2. Evaluate the agent
    print("Starting evaluation :")
    avg_score, eval_rewards = evaluate(Q_learned, n_episodes=10, render=True)
    
    print(f"Average evaluation score: {avg_score}\nEvaluation finished.")

    print("Plotting evaluation rewards... \n")
    plt.figure()
    plt.plot(eval_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Evaluation Rewards")
    plt.show()
