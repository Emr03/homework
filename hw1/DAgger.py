import numpy as np
import tensorflow as tf
import gym
import load_policy
from model import Model
from ant_model import AntModel
from humanoid_model import HumanoidModel
from behavior_cloning import BehaviorCloning

class DAgger:

    def __init__(self, behavior_cloner, expert_policy_file):

        self.behavior_cloner = behavior_cloner

        print('loading and building expert policy')
        self.policy_fn = load_policy.load_policy(expert_policy_file)
        print('loaded and built')

    def ask_expert(self, observations):

        new_actions = self.policy_fn(observations)
        new_actions = new_actions.reshape(new_actions.shape[0], new_actions.shape[-1])
        self.behavior_cloner.add_data(observations, new_actions)

    def run(self, n_iter=4, n_epochs=300, n_steps=500):

        self.behavior_cloner.train(n_epochs)
        returns_list = []

        for i in range(n_iter):

            total_return, new_observations = self.behavior_cloner.test(n_steps)
            self.ask_expert(new_observations)
            self.behavior_cloner.train(n_epochs)
            returns_list.append(total_return)

        total_return, obs = self.behavior_cloner.test(1000)
        returns_list.append(total_return)
        print(returns_list)

if __name__ == "__main__":

    obs_file = '/home/elsa/Desktop/homework/hw1/Humanoid-v1_obs.npy'
    actions_file = '/home/elsa/Desktop/homework/hw1/Humanoid-v1_actions.npy'

    copyCat = BehaviorCloning('Humanoid-v1', HumanoidModel, obs_file, actions_file, batch_size=100)
    copyCat.model.load("humanoid_model.ckpt")
    dAgger = DAgger(copyCat, 'experts/Humanoid-v1.pkl')

    dAgger.run()

