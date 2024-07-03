import gym
import random
import numpy as np


class RandomActionWrapper(gym.ActionWrapper):
    """Wrapper for the environment that will randomly select an action with a given probability.
    Args:
        env: gym.Env - Environment to wrap
        epsilon: float - Probability of selecting a random action
    """

    def __init__(self, env, epsilon=0.1):
        """Initialize the wrapper.
        Args:
            env: gym.Env - Environment to wrap
            epsilon: float - Probability of selecting a random action
        """
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    def action(self, action):
        """Select an action to perform.
        Args:
            action: int - Action to perform
        Returns: int - Action to perform
        """
        if random.random() < self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action


def run_experiment(env_name="CartPole-v1"):
    """Run an experiment on the environment.
    Args:
        env_name: str - Name of the environment to run the experiment on
    Returns: float - Total reward obtained during the experiment
    """
    env = RandomActionWrapper(gym.make(env_name))  # Wrapping the environment
    env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs, reward, done, _, _ = env.step(0)
        total_reward += reward
        total_steps += 1
        if done:
            break
    print(f"Total reward: {total_reward}; Steps done: {total_steps}")
    return total_reward


if __name__ == "__main__":
    results = []
    for _ in range(1000):
        results.append(run_experiment())
    print("–" * 10)
    print("Random action wrapper results:")
    print(f"Minimum reward: {np.min(results)}")
    print(f"Maximum reward: {np.max(results)}")
    print(f"Mean reward: {np.mean(results)}")
    print(f"Median reward: {np.median(results)}")
