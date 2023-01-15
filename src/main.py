import sys
import gym_environments
import gym
import numpy as np
from agent import TwoArmedBandit

if __name__ == '__main__':
    num_iterations = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    version = 'v0' if len(sys.argv) < 3 else sys.argv[2]
    methodIDs = ['random', 'greedy', 'epsilon-greedy']
    alpha = 0.1
    env = gym.make(f'TwoArmedBandit-{version}')
    agent = TwoArmedBandit(alpha) 

    env.reset(options={'delay': 1})

    for methodID in methodIDs:
        for alpha in np.arange(0.1, 1.1, 0.1):
            totalReward = 0
            for iteration in range(num_iterations):
                action = agent.get_action(methodID)    
                _, reward, _, _, _ = env.step(action)
                agent.update(action, reward)
                # agent.render()
                totalReward += reward
            print(f'Alpha: {alpha:<.1f}|Action: {methodID}|Reward:{totalReward}')
    env.close()