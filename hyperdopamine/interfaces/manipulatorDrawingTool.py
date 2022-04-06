import functools
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib import animation as animation
from matplotlib.widgets import Button

from manipulator_pair import manipulator_pair


class ManipulatorDrawingTool:

    def __init__(self, graphSize, links, output):

        self.lines1 = []
        self.lines2 = []

        self.legend = ['current position', 'desired position']

        self.manipulator_pair = manipulator_pair(links, output)

        self.axes_labels = [[[] for _ in range(6)] for _ in range(2)]
        self.axes_buttons = [[[] for _ in range(6)] for _ in range(2)]
        self.title_labels = [[] for _ in range(2)]
        self.title_buttons = [[] for _ in range(2)]
        self.buttons = [[[[] for _ in range(2)] for _ in range(2)] for _ in range(6)]
        self.axes_theta = [[[[] for _ in range(2)] for _ in range(2)] for _ in range(6)]
        self.axes_deltas = [[] for _ in range(3)]
        self.deltaButtons = [[] for _ in range(3)]
        self.fig = plt.figure()
        self.fig.set_size_inches(10, 8)
        self.ax = p3.Axes3D(self.fig, rect=(-0.1, 0.15, 0.9, 0.9))
        self.ax.set_xlim3d([-graphSize, graphSize])
        self.ax.set_xlabel('X')
        self.ax.set_ylim3d([-graphSize, graphSize])
        self.ax.set_ylabel('Y')
        self.ax.set_zlim3d([-graphSize, graphSize])
        self.ax.set_zlabel('Z')
        self.ax.set_title('3D Test')

        self.outputCart = output[0]
        self.outputPolar = output[1]
        self.outputQuat = output[2]

    def draw(self, points, save, show):

        x_size, y_size, z_size = points.shape
        lines = self.ax.plot(points[0, :, 0], points[1, :, 0], points[2, :, 0])[0]
        line_ani = animation.FuncAnimation(self.fig, update_lines, z_size, fargs=(points, lines),
                                           interval=200, blit=False)
        if save:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y-%H-%M-%S.gif")
            path = pathlib.Path(__file__).parent.absolute()
            path = path.joinpath('gifs')
            path = path.joinpath('manipulator_' + dt_string)
            dt = str(path)
            writergif = animation.PillowWriter(fps=30)
            line_ani.save(dt, writer=writergif)
        if show:
            plt.show(block=True)

    def interactive_draw(self):
        self.setup_buttons()
        self.lines1 = self.ax.plot(self.manipulator_pair.manipulator[0].position[0, :],
                                   self.manipulator_pair.manipulator[0].position[1, :],
                                   self.manipulator_pair.manipulator[0].position[2, :],
                                   label=self.legend[0])[0]

        self.lines2 = self.ax.plot(self.manipulator_pair.manipulator[0].position[0, :],
                                   self.manipulator_pair.manipulator[0].position[1, :],
                                   self.manipulator_pair.manipulator[0].position[2, :],
                                   label=self.legend[0])[0]

        plt.legend([self.lines1, self.lines2], self.legend, loc='best', bbox_to_anchor=(0.5, 1.5, 0.5, 0.5))
        plt.show(block=True)

    def change_angle(self, event, value):
        diff = 10
        theta, sign, arm = value
        delta = np.zeros([6, 1])
        delta[theta] = (2 * sign - 1) * diff
        self.manipulator_pair.update_angles(arm, delta)
        self.axes_buttons[arm][theta].label.set_text(str(self.manipulator_pair.manipulator[arm].angles[theta, 0]))
        if arm:
            self.lines2.set_data_3d(self.manipulator_pair.manipulator[1].position[0, :], \
                                    self.manipulator_pair.manipulator[1].position[1, :], \
                                    self.manipulator_pair.manipulator[1].position[2, :])
        else:
            self.lines1.set_data_3d(self.manipulator_pair.manipulator[0].position[0, :], \
                                    self.manipulator_pair.manipulator[0].position[1, :], \
                                    self.manipulator_pair.manipulator[0].position[2, :])
        self.updateDeltas()

    def setup_buttons(self):
        posX = [0.8, 0.9]
        posY = [0.85, 0.4]
        for arm in range(2):
            self.title_labels[arm] = plt.axes([0.8, posY[arm] + 0.05, 0.15, 0.045])
            self.title_buttons[arm] = Button(self.title_labels[arm], self.legend[arm])
        for theta in range(6):
            for arm in range(2):
                for sign in range(2):
                    self.axes_theta[theta][sign][arm] = (
                        plt.axes([posX[sign], posY[arm] - 0.06 * theta, 0.05, 0.045]))
                    name = ('-' if sign == 0 else '') + 'T' + str(theta + 1)
                    self.buttons[theta][sign][arm] = Button(self.axes_theta[theta][sign][arm], name)
                    self.buttons[theta][sign][arm].on_clicked(
                        functools.partial(self.change_angle, value=(theta, sign, arm)))
                self.axes_labels[arm][theta] = plt.axes([0.85, posY[arm] - 0.06 * theta, 0.05, 0.045])
                self.axes_buttons[arm][theta] = Button(self.axes_labels[arm][theta], '0')
        self.axes_deltas[0] = plt.axes([0.01, 0.01, 0.25, 0.1])
        self.axes_deltas[1] = plt.axes([0.26, 0.01, 0.25, 0.1])
        self.axes_deltas[2] = plt.axes([0.51, 0.01, 0.25, 0.1])

        if self.outputCart:  self.deltaButtons[0] = Button(self.axes_deltas[0], f'Cartesian delta:\nX: 0\nY: 0\nZ: 0')
        if self.outputPolar: self.deltaButtons[1] = Button(self.axes_deltas[1], f'Polar delta:\nRadius: 0\nAzimuth: 0\nElevation: 0')
        if self.outputQuat:  self.deltaButtons[2] = Button(self.axes_deltas[2], f'Quaternion delta:\n1\n0i\n0j\n0k')

    def updateDeltas(self):
        cart, polar, quat = self.manipulator_pair.calculateDifference()
        if self.outputCart:
            self.deltaButtons[0].label.set_text(
                f'Cartesian delta:\nX: {cart[0, 0]}\nY: {cart[1, 0]}\nZ: {cart[2, 0]}')
        if self.outputPolar:
            self.deltaButtons[1].label.set_text(
                f'Polar delta:\nRadius: {polar[0, 0]}\nAzimuth: {polar[1, 0]}\nElevation: {polar[2, 0]}')
        if self.outputQuat:
            self.deltaButtons[2].label.set_text(
                f'Quaternion delta:\n{quat[0, 0]}\n{quat[1, 0]}i\n{quat[2, 0]}j\n{quat[3, 0]}k')


def update_lines(num, dataLines, lines):
    lines.set_data(dataLines[0:2, :, num])
    lines.set_3d_properties(dataLines[2, :, num])
    return lines
