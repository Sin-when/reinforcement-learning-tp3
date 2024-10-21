"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import matplotlib.pyplot as plt
import typing as t
import gymnasium as gym
import numpy as np
from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from gym.wrappers import RecordVideo


#################################################
# 0. Utils functions 
#################################################


env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


# Function to plot rewards
def plot_rewards(rewards, algorithm_name):
    plt.figure(figsize=(10, 6))

    # Calcuate the moving averages
    mean_rewards = np.convolve(rewards, np.ones(10) / 10, mode='valid')

    # Plot the mean rewards
    plt.plot(mean_rewards, label=f"{algorithm_name} (mean over 10 episodes)")
    plt.xlabel('Episodes')
    plt.ylabel('Average Total Reward')
    plt.title("Reward Progression")
    plt.legend()
    
    plt.savefig(algorithm_name + "_rewards.png")
    plt.close()  


# Function to launch the simulation
def launch_simulation(agent, env):
    rewards_qlearning = []

    ploting_rewards = []

    
    best_mean_reward = -float("inf")
    stocking_agent = agent


    for i in range(6000):
        reward = play_and_train(env, agent)
        rewards_qlearning.append(reward)

        if i % 200 == 0:
            current_mean = np.mean(rewards_qlearning[-100:])
            ploting_rewards.append(current_mean)
            print("mean reward", current_mean)

        if current_mean > best_mean_reward:
            best_mean_reward = current_mean
            stocking_agent = agent
        else:
            agent = stocking_agent

    ploting_rewards.append(np.mean(rewards_qlearning[-100:]))

    env.close()  
    return rewards_qlearning, ploting_rewards
        
    

#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.1, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for _ in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)
        # BEGIN SOLUTION
        agent.update(s, a, r, next_s)
        total_reward += r
        s = next_s

        if done:
            break
        # END SOLUTION

    return total_reward



print("Q-Learning Algorithm")
print("")
print("")



#env = RecordVideo(env, video_folder="./Q-Learning", episode_trigger=lambda x: x % 2000 == 0) 
#rewards_qlearning, ploting_rewards = launch_simulation(agent, env)
#plot_rewards(ploting_rewards, "Q-Learning")

#assert np.mean(rewards_qlearning[-100:]) > 0.0


# TODO: créer des vidéos de l'agent en action

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


print("Q-Learning with Epsilon Scheduling Algorithm")
print("")
print("")


# env = RecordVideo(env, video_folder="./Q-Learning-Eps", episode_trigger=lambda x: x % 1000 == 0) 

agent = QLearningAgentEpsScheduling(
    learning_rate=0.1, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

env = RecordVideo(env, video_folder="./Q-Learning-Eps", episode_trigger=lambda x: x % 2000 == 0) 
rewards_qlearning, ploting_rewards = launch_simulation(agent, env)
plot_rewards(ploting_rewards, "Q-Learning-Eps")

assert np.mean(rewards_qlearning[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action


####################
# 3. Play with SARSA
####################

print("Q-Learning with Sarsa")
print("")
print("")



agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

env = RecordVideo(env, video_folder="./SARSA", episode_trigger=lambda x: x % 2000 == 0) 
rewards_qlearning, ploting_rewards = launch_simulation(agent, env)
plot_rewards(ploting_rewards, "SARSA")