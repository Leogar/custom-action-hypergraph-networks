import gym
import numpy as np
import tensorflow as tf
from hyperdopamine.interfaces.manipulator_pair import manipulator_pair


class myEnv(gym.Env):

    def __init__(self):
        links = [0, 30, 20, 10, 0, 5]
        self.used_coords = [1, 0, 1]  # cartesian spherical quaternion
        self.manipulators = manipulator_pair(links, self.used_coords)
        cartesianLowBoundary = [-65, -65, -65]
        cartesianHighBoundary = [65, 65, 65]
        SphericalLowBoundary = [-65, -180, -180]
        SphericalHighBoundary = [65, 180, 180]
        QuatLowBoundary = [-1, -1, -1, -1]
        QuatHighBoundary = [1, 1, 1, 1]
        lowBoundary = []
        highBoundary = []
        self.best_difference = []
        if self.used_coords[0]:
            lowBoundary = lowBoundary + cartesianLowBoundary
            highBoundary = highBoundary + cartesianHighBoundary
            self.best_difference = self.best_difference + [0, 0, 0]
        if self.used_coords[1]:
            lowBoundary = lowBoundary + SphericalLowBoundary
            highBoundary = highBoundary + SphericalHighBoundary
            self.best_difference = self.best_difference + [0, 0, 0]
        if self.used_coords[2]:
            lowBoundary = lowBoundary + QuatLowBoundary
            highBoundary = highBoundary + QuatHighBoundary
            self.best_difference = self.best_difference + [1, 0, 0, 0]


        self.best_difference = np.array(self.best_difference)
        self.best_difference = self.best_difference.reshape(len(self.best_difference), 1)
        #  lowBoundary = np.array([-65, -65, -65, -65, -180, -180, -1, -1, -1, -1])
        #  highBoundary = np.array([65, 65, 65, 65, 180, 180, 1, 1, 1, 1])

        self.action_space = gym.spaces.Box(np.full((6, 1), -2), np.full((6, 1), 2))
        self.observation_space = gym.spaces.Box(np.array(lowBoundary), np.array(highBoundary))

        self.milestone = 1

    def step(self, action):
        # nest_difference = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        self.manipulators.update_angles(1, action)

        state = self.manipulators.calculateDifference()
        state_arr = state.reshape((7, 1))
        res = (state_arr - self.best_difference) ** 2
        diff = sum(res)
        reward = (self.previous_diff - diff) * 10
        reward = reward[0]
        self.previous_diff = diff
        #reward = -diff
        #reward = np.log(diff)
        #reward = reward[0]
        done = diff < 1e-3
        #if diff < 10 ** (4 - self.milestone):
        #    add = 10 ** self.milestone
        #    reward = reward + add
        #    self.milestone = self.milestone + 1
        #    #tf.compat.v1.logging.info('\t Milestone %s reached! Reward deployed: %s', str(self.milestone), str(add))
        if done:
            reward += 1e15
        return state_arr, reward, 0, 0

    def reset(self):
        self.manipulators.randomize()
        state = self.manipulators.calculateDifference()
        state_arr = state.reshape((7, 1))
        res = (state_arr - self.best_difference) ** 2
        self.previous_diff = sum(res)

        self.milestone = 0
        return state_arr
