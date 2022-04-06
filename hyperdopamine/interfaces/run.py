# This script modifies Dopamine with the original copyright note:
# 
# Copyright 2018 The Dopamine Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags

from hyperdopamine.interfaces import run_model_test
from hyperdopamine.interfaces.dof_manipulator import dof_6_manipulator
from hyperdopamine.interfaces.manipulator_pair import manipulator_pair
from hyperdopamine.interfaces.manipulatorDrawingTool import ManipulatorDrawingTool
import tensorflow as tf

flags.DEFINE_string('base_dir', None,
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string('schedule', None,
                    'Schedule of whether to train and evaluate or just train.')
flags.DEFINE_multi_string(
    'gin_files', [], 'List of paths to gin configuration files (e.g.'
                     '"hyperdopamine/agents/dqn/dqn.gin").')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files '
    '(e.g. "DQNAgent.epsilon_train=0.1",'
    '      "create_environment.game_name="Pong"").')

FLAGS = flags.FLAGS


def main(unused_argv):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    run_model_test.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = run_model_test.create_runner(FLAGS.base_dir, FLAGS.schedule)

    links = [0, 30, 20, 10, 0, 5]
    pair = manipulator_pair(links, [1, 0, 1])
    plotter = ManipulatorDrawingTool(50, links)
    # plotter.interactive_draw()

    pair.randomize()
    points = pair.get_current_points()
    diff = pair.calculateDifference()
    dff = np.concatenate(diff)
    dff_array = dff
    i = 0
    done = 0
    best_difference = np.array([0, 0, 0,  1, 0, 0, 0]) # 0, 0, 0,
    best_difference = best_difference.reshape(7, 1) # 10
    while i < 1000 and done == 0:
        diff = pair.calculateDifference()
        diff_arr = np.concatenate(diff)
        action = runner._agent.step(0, diff_arr) - 1
        pair.update_angles(1, action)
        points = np.append(points, pair.get_current_points(), 2)
        dff_array = np.append(dff_array, np.concatenate(diff), 1)
        if sum((diff_arr - best_difference) ** 2) < 0.01:
            done = 1
        i += 1
        print(i, end="\r")
    # plotter.draw(points, dff_array, save=True, show=False)
    plotter.draw(points, dff_array, save=False, show=True)
    # plotter.interactive_draw()
    plt.figure(3)
    MSE = np.sum(dff_array ** 2, 0)
    plt.plot(range(len(MSE)), MSE)
    plt.show()


if __name__ == '__main__':
    flags.mark_flag_as_required('base_dir')
    app.run(main)
