import gym
import gym.spaces
from random import *
env = gym.make('FrozenLake-v0')
#info for 4x4 grid
print(env.observation_space)

print(env.action_space) #up, down, left, right
rand = randint(1, 2)

score = 0
for _ in range(10000):
    env.reset()  # resets
    for t in range(1000):

        action = env.action_space.sample()
        observation, reward, done, info = env.step(2)

        env.render()
        if done:
            score += reward

            break
print(score) 