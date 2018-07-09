import numpy as np
import tensorflow as tf
import gym
from model import Model
from ant_model import AntModel
from humanoid_model import HumanoidModel

class BehaviorCloning:

    def __init__(self, env_name, model, obs_file, actions_file, batch_size):
        """

        :param env_name: name of the environment to train on
        :param model: subclass of Model
        :param obs_file: npy file of expert observations
        :param actions_file: npy file of expert actions
        :param batch_size: used for training
        """

        self.env_name = env_name
        self.observations = np.load(obs_file)
        print(self.observations.shape)
        #self.observations = np.reshape(self.observations, self.observations.shape[0], 1, self.observations.shape[-1])
        self.actions = np.load(actions_file)
        self.actions = self.actions.reshape(self.actions.shape[0], self.actions.shape[-1])
        print(self.actions.shape)

        # instantiate and build the model
        self.model = model(batch_size=batch_size, x=self.observations, y=self.actions)
        self.model.build()

    def add_data(self, observations, actions):

        self.observations = np.concatenate((self.observations, observations))
        self.actions = np.concatenate((self.actions, actions))
        self.model.set_data(self.observations, self.actions)

    def train(self, n_epochs):

        self.model.train(n_epochs)

    def test(self, time_steps):

        # test the model, generate and return new rollout
        env = gym.make(self.env_name)
        total_return = 0
        observation = env.reset()
        observations = []

        for t in range(time_steps):

            observation = observation.reshape(1, observation.size)
            env.render()

            # compute action
            action = self.model.predict(observation)

            observation, reward, done, info = env.step(action)  # take a random action
            observations.append(observation)

            total_return += reward

        print(total_return)

        return total_return, np.array(observations)


if __name__ == "__main__":

    # obs_file = '/home/elsa/Desktop/homework/hw1/Ant-v1_obs.npy'
    # actions_file = '/home/elsa/Desktop/homework/hw1/Ant-v1_actions.npy'
    #
    # copyCat = BehaviorCloning('Ant-v1', AntModel, obs_file, actions_file, batch_size=100)
    #
    # #copyCat.train(n_epochs=2000)
    # copyCat.model.load("ant_model.ckpt")
    #
    # copyCat.test(time_steps=1000)

    obs_file = '/home/elsa/Desktop/homework/hw1/Humanoid-v1_obs.npy'
    actions_file = '/home/elsa/Desktop/homework/hw1/Humanoid-v1_actions.npy'

    copyCat = BehaviorCloning('Humanoid-v1', HumanoidModel, obs_file, actions_file, batch_size=100)

    copyCat.train(n_epochs=800)
    #copyCat.model.load("_model.ckpt")

    copyCat.test(time_steps=1000)

