import numpy as np


class dof_6_manipulator:

    def __init__(self, links):
        self.links = links
        self.angles = np.zeros([6, 1])
        self.endEffectorPos = 0
        self.position, self.orientation = self.calculate_forward_kinematics(self.angles)
        self.endEffectorPos = self.position[:, -1]

    def generate_points(self, start, stop, step):
        angles = range(start, stop + 1, step)
        points = np.zeros([3, 7, (int((stop - start) / step) + 1) ** 5])
        pointer = 0
        for i in angles:
            for j in angles:
                print(i, j)
                for k in angles:
                    for l in angles:
                        for m in angles:
                            p, o = self.calculate_forward_kinematics([i, j, k, l, m, 0])
                            points[:, :, pointer] = p
                            pointer += 1
        return points

    def update_angles(self, angles_delta):
        self.angles += angles_delta
        self.position, self.orientation = self.calculate_forward_kinematics(self.angles)
        self.endEffectorPos = self.position[:, -1]
        return self.position, self.orientation

    def calculate_forward_kinematics(self, angles):
        angles = (np.asarray(angles) * np.pi / 180).flatten()
        s10 = self.links[4] * np.cos(angles[3]) * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(
            angles[4])
        s11 = self.links[4] * np.cos(angles[1]) * np.cos(angles[2]) * np.cos(angles[3]) * np.sin(angles[0]) * np.sin(
            angles[4])
        s12 = self.links[1] * np.sin(angles[0]) * np.sin(angles[1])
        s13 = self.links[4] * np.sin(angles[1] + angles[2]) * np.cos(angles[4]) * np.sin(angles[0])
        s14 = self.links[4] * np.cos(angles[0]) * np.sin(angles[3]) * np.sin(angles[4])
        s15 = self.links[3] * np.sin(angles[1] + angles[2]) * np.sin(angles[0])
        s16 = self.links[2] * np.sin(angles[1] + angles[2]) * np.sin(angles[0])
        s17 = self.links[4] * np.cos(angles[0]) * np.cos(angles[3]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(
            angles[4])
        s18 = self.links[4] * np.cos(angles[0]) * np.cos(angles[1]) * np.cos(angles[2]) * np.cos(angles[3]) * np.sin(
            angles[4])
        s19 = self.links[1] * np.cos(angles[0]) * np.sin(angles[1])
        s20 = self.links[4] * np.sin(angles[1] + angles[2]) * np.cos(angles[0]) * np.cos(angles[4])
        s21 = self.links[4] * np.sin(angles[0]) * np.sin(angles[3]) * np.sin(angles[4])
        s22 = self.links[3] * np.sin(angles[1] + angles[2]) * np.cos(angles[0])
        s23 = self.links[2] * np.sin(angles[1] + angles[2]) * np.cos(angles[0])
        s24 = self.links[2] * np.cos(angles[1] + angles[2])
        s25 = (self.links[4] * np.sin(angles[1] + angles[2]) * np.sin(angles[3] - angles[4])) / 2
        s26 = (self.links[4] * np.sin(angles[1] + angles[2]) * np.sin(angles[3] + angles[4])) / 2
        s27 = self.links[3] * np.cos(angles[1] + angles[2])
        s28 = self.links[4] * np.cos(angles[1] + angles[2]) * np.cos(angles[4])
        s29 = self.links[2] * np.sin(angles[1] + angles[2])
        s1 = np.sin(angles[1] + angles[2]) * np.sin(angles[0]) * np.sin(angles[4]) - np.cos(angles[0]) * np.cos(
            angles[4]) * np.sin(angles[3]) - np.cos(angles[1]) * np.cos(angles[2]) * np.cos(angles[3]) * np.cos(
            angles[4]) * np.sin(angles[0]) + np.cos(angles[3]) * np.cos(angles[4]) * np.sin(angles[0]) * np.sin(
            angles[1]) * np.sin(angles[2])
        s2 = np.sin(angles[1] + angles[2]) * np.cos(angles[0]) * np.sin(angles[4]) + np.cos(angles[4]) * np.sin(
            angles[0]) * np.sin(angles[3]) + np.cos(angles[0]) * np.cos(angles[3]) * np.cos(angles[4]) * np.sin(
            angles[1]) * np.sin(angles[2]) - np.cos(angles[0]) * np.cos(angles[1]) * np.cos(angles[2]) * np.cos(
            angles[3]) * np.cos(angles[4])
        s3 = np.cos(angles[0]) * np.cos(angles[3]) - np.cos(angles[1]) * np.cos(angles[2]) * np.sin(angles[0]) * np.sin(
            angles[3]) + np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3])
        s4 = np.cos(angles[3]) * np.sin(angles[0]) + np.cos(angles[0]) * np.cos(angles[1]) * np.cos(angles[2]) * np.sin(
            angles[3]) - np.cos(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[3])
        s5 = s10 - s15 - s12 - s14 - self.links[5] * np.cos(angles[0]) * np.sin(angles[3]) * np.sin(angles[4]) - s13 - \
             self.links[
                 5] * np.sin(angles[1] + angles[2]) * np.cos(angles[4]) * np.sin(angles[0]) - s11 - self.links[
                 5] * np.cos(
            angles[1]) * np.cos(angles[2]) * np.cos(angles[3]) * np.sin(angles[0]) * np.sin(angles[4]) - s16 + \
             self.links[
                 5] * np.cos(angles[3]) * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[4])
        s6 = s21 - s23 - s22 - s19 + self.links[5] * np.sin(angles[0]) * np.sin(angles[3]) * np.sin(angles[4]) - s20 - \
             self.links[
                 5] * np.sin(angles[1] + angles[2]) * np.cos(angles[4]) * np.cos(angles[0]) - s18 - self.links[
                 5] * np.cos(
            angles[0]) * np.cos(angles[1]) * np.cos(angles[2]) * np.cos(angles[3]) * np.sin(angles[4]) + s17 + \
             self.links[
                 5] * np.cos(angles[0]) * np.cos(angles[3]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(angles[4])
        s7 = self.links[0] + s24 + s27 + self.links[1] * np.cos(angles[1]) + s25 + (
                self.links[5] * np.sin(angles[1] + angles[2]) * np.sin(angles[3] - angles[4])) / 2 - s26 - (
                     self.links[5] * np.sin(angles[1] + angles[2]) * np.sin(angles[3] + angles[4])) / 2 + s28 + \
             self.links[
                 5] * np.cos(angles[1] + angles[2]) * np.cos(angles[4])
        s8 = s29 + self.links[3] * np.sin(angles[1] + angles[2]) + self.links[1] * np.sin(angles[1])
        s9 = s29 + self.links[1] * np.sin(angles[1])
        points = np.zeros([3, 7])

        points[2, 1] = self.links[0]
        points[:, 2] = np.array([-s19, -s12, self.links[0] + self.links[1] * np.cos(angles[1])])
        points[:, 3] = np.array(
            [-np.cos(angles[0]) * s9, -np.sin(angles[0]) * s9, self.links[0] + s24 + self.links[1] * np.cos(angles[1])])
        points[:, 4] = np.array(
            [-np.cos(angles[0]) * s8, -np.sin(angles[0]) * s8,
             self.links[0] + s24 + s27 + self.links[1] * np.cos(angles[1])])
        points[:, 5] = np.array([s21 - s23 - s22 - s19 - s20 - s18 + s17, s10 - s15 - s12 - s14 - s13 - s11 - s16,
                                 self.links[0] + s24 + s27 + self.links[1] * np.cos(angles[1]) + s25 - s26 + s28])
        points[:, 6] = np.array([s6, s5, s7])

        orientation = np.array([[-np.sin(angles[5]) * s4 - np.cos(angles[5]) * s2,
                                 np.sin(angles[5]) * s2 - np.cos(angles[5]) * s4,

                                 np.sin(angles[0]) * np.sin(angles[3]) * np.sin(angles[4]) - np.sin(
                                     angles[1] + angles[2]) * np.cos(angles[0]) * np.cos(angles[4]) - np.cos(
                                     angles[0]) * np.cos(angles[1]) * np.cos(angles[2]) * np.cos(angles[3]) * np.sin(
                                     angles[4]) + np.cos(
                                     angles[0]) * np.cos(angles[3]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(
                                     angles[4])],

                                [np.sin(angles[5]) * s3 - np.cos(angles[5]) * s1,
                                 np.cos(angles[5]) * s3 + np.sin(angles[5]) * s1,

                                 np.cos(angles[3]) * np.sin(angles[0]) * np.sin(angles[1]) * np.sin(angles[2]) * np.sin(
                                     angles[4]) - np.cos(angles[0]) * np.sin(angles[3]) * np.sin(angles[4]) - np.cos(
                                     angles[1]) * np.cos(angles[2]) * np.cos(angles[3]) * np.sin(angles[0]) * np.sin(
                                     angles[4]) - np.sin(angles[1] + angles[2]) * np.cos(angles[4]) * np.sin(
                                     angles[0])],

                                [np.cos(angles[1] + angles[2]) * np.cos(angles[5]) * np.sin(angles[4]) - np.sin(
                                    angles[1] + angles[2]) * np.sin(angles[3]) * np.sin(angles[5]) + np.sin(
                                    angles[1] + angles[2]) * np.cos(angles[3]) * np.cos(angles[4]) * np.cos(angles[5]),

                                 -np.sin(angles[1] + angles[2]) * np.cos(angles[5]) * np.sin(angles[3]) - np.cos(
                                     angles[1] + angles[2]) * np.sin(angles[4]) * np.sin(angles[5]) - np.sin(
                                     angles[1] + angles[2]) * np.cos(angles[3]) * np.cos(angles[4]) * np.sin(angles[5]),

                                 np.cos(angles[1] + angles[2]) * np.cos(angles[4]) - np.sin(
                                     angles[1] + angles[2]) * np.cos(
                                     angles[3]) * np.sin(angles[4])]])
        return points, orientation
