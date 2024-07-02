import gym
import numpy as np

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    total_reward = 0
    total_steps = 0
    obs = env.reset()
    while True:
        action = env.action_space.sample()  # get random sample from action space
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print(f"Total reward: {total_reward}; Steps done: {total_steps}")
