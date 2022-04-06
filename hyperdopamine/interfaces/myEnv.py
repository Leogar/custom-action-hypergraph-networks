import gym
import numpy as np
import tensorflow as tf
from hyperdopamine.interfaces.manipulator_pair import manipulator_pair


class myEnv(gym.Env):

    def __init__(self):
        links = [0, 30, 20, 10, 0, 5]
        self.manipulators = manipulator_pair(links)
        self.used_coords = [1, 0, 1]  # cartesian spherical quaternion
        cartesianLowBoundary = [-65, -65, -65]
        cartesianHighBoundary = [65, 65, 65]
        SphericalLowBoundary = [-65, -180, -180]
        SphericalHighBoundary = [65, 180, 180]
        QuatLowBoundary = [-1, -1, -1, -1]
        QuatHighBoundary = [1, 1, 1, 1]
        lowBoundary = []
        highBoundary = []
        if self.used_coords[0]:
            lowBoundary = lowBoundary + cartesianLowBoundary
            highBoundary = highBoundary + cartesianHighBoundary
        if self.used_coords[1]:
            lowBoundary = lowBoundary + SphericalLowBoundary
            highBoundary = highBoundary + SphericalHighBoundary
        if self.used_coords[2]:
            lowBoundary = lowBoundary + QuatLowBoundary
            highBoundary = highBoundary + QuatHighBoundary

        #  lowBoundary = np.array([-65, -65, -65, -65, -180, -180, -1, -1, -1, -1])
        #  highBoundary = np.array([65, 65, 65, 65, 180, 180, 1, 1, 1, 1])

        self.action_space = gym.spaces.Box(np.full((6, 1), -2), np.full((6, 1), 2))
        self.observation_space = gym.spaces.Box(np.array(lowBoundary), np.array(highBoundary))

        self.milestone = 0

    def step(self, action):
        best_difference = []
        if self.used_coords[0]:
            best_difference = best_difference + [0, 0, 0]
        if self.used_coords[1]:
            best_difference = best_difference + [0, 0, 0]
        if self.used_coords[2]:
            best_difference = best_difference + [1, 0, 0, 0]
        # nest_difference = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        best_difference = np.array(best_difference)
        best_difference = best_difference.reshape(len(best_difference), 1)
        self.manipulators.update_angles(1, action)

        state = self.manipulators.calculateDifference(self.used_coords)
        state_arr = np.concatenate(state)
        res = (state_arr - best_difference) ** 2
        diff = sum(res)
        reward = -diff
        reward = reward[0]
        done = diff < 0.001
        if diff < 10 ** (4 - self.milestone):
            add = 10 ** self.milestone
            reward = reward + add
            self.milestone = self.milestone + 1
            tf.compat.v1.logging.info('\t Milestone %s reached! Reward deployed: %s', str(self.milestone), str(add))
        if done:
            reward += 1000000
        return state_arr, reward, 0, 0

    def reset(self):
        self.manipulators.randomize()
        state = self.manipulators.calculateDifference(self.used_coords)
        state_arr = np.concatenate(state)
        self.milestone = 0
        return state_arr
