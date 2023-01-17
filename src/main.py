import sys
import gym_environments
import gym
import numpy as np
import csv
from agent import TwoArmedBandit

if __name__ == '__main__':
    num_iterations = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    version = 'v0' if len(sys.argv) < 3 else sys.argv[2]
    methodIDs = ['random', 'greedy', 'epsilon-greedy']
    #Indicates the trust level for the actions
    alpha = 0.1
    #Sets a threshold for how willing is the agent to explore
    epsilon = 0.1
    env = gym.make(f'TwoArmedBandit-{version}')

    for methodID in methodIDs:
        env.reset(options={'delay': 1})
        for epsilon in np.arange(0.1, 1.1, 0.1):
            for alpha in np.arange(0.1, 1.1, 0.1):
                agent = TwoArmedBandit(alpha, epsilon)
                totalReward = 0
                for iteration in range(num_iterations):
                    action = agent.get_action(methodID)    
                    _, reward, _, _, _ = env.step(action)
                    agent.update(action, reward)
                    totalReward += reward     
                print(f'Steps: {num_iterations} | Alpha: {alpha:<.1f} | Epsilon:{epsilon::<.1f} | Action: {methodID} | Reward:{totalReward}')
        print('\n')
        env.close()