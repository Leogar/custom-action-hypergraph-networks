import numpy as np
from pyquaternion import Quaternion

from dof_manipulator import dof_6_manipulator


class manipulator_pair:

    def __init__(self, links, output):
        self.decimalsToRound = 4
        self.manipulator = [dof_6_manipulator(links), dof_6_manipulator(links)]

        self.outputCart = output[0]
        self.outputPolar = output[1]
        self.outputQuat = output[2]

    def update_angles(self, manipulator_number, angles_delta):
        self.manipulator[manipulator_number].update_angles(angles_delta)

    def calculateDifference(self):
        endEffectorPos = np.array([self.manipulator[0].endEffectorPos, self.manipulator[1].endEffectorPos])
        cartesianDifference = np.diff(endEffectorPos, axis=0) if self.outputCart else 0
        polarDifference = np.diff(np.array([
            np.sqrt(np.sum(endEffectorPos ** 2, axis=1)),
            np.arctan2(endEffectorPos[:, 1], endEffectorPos[:, 0]) * 180 / np.pi,
            np.arctan(np.sqrt(np.sum(endEffectorPos[:, :-1] ** 2, axis=1)) / endEffectorPos[:, -1]) * 180 / np.pi
        ]), axis=1) if self.outputPolar else 0
        if self.outputQuat:
            Q1 = Quaternion(matrix=self.manipulator[0].orientation)
            Q2 = Quaternion(matrix=self.manipulator[1].orientation)
            quaternionDifference = Q2 * Q1.inverse
        else:
            quaternionDifference = 0

        return np.around(cartesianDifference.reshape(-1, 1), self.decimalsToRound) if self.outputCart else 0, \
               np.around(polarDifference.reshape(-1, 1), self.decimalsToRound) if self.outputPolar else 0, \
               np.around(quaternionDifference.elements.reshape(-1, 1) if self.outputQuat else 0, self.decimalsToRound)


def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
                     x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
                     -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
                     x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0], dtype=np.float64)
