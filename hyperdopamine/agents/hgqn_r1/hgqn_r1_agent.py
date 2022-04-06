from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from hyperdopamine.agents.dqn import dqn_agent
from hyperdopamine.agents.bdqn import bdqn_agent
from hyperdopamine.agents import networks
import numpy as np
import tensorflow as tf
import gin.tf

slim = tf.contrib.slim


@gin.configurable
class HGQNr1Agent(bdqn_agent.BDQNAgent):
  ''' An implementation of the rank-1 monotonic-mixer HGQN agent. '''

  def __init__(self,
               sess,
               num_sub_actions,
               observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE,
               observation_dtype=dqn_agent.NATURE_DQN_DTYPE,
               stack_size=dqn_agent.NATURE_DQN_STACK_SIZE,
               network=networks.branching_network,
               mixing_network=networks.sum_mixer,
               use_dueling=False,
               double_dqn=False,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=10000,
               update_period=1,
               target_update_period=2000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.05,
               epsilon_eval=0.001,
               epsilon_decay_period=50000,
               replay_scheme='uniform',
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=4,
               loss_type='MSE',
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00001, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):
    assert isinstance(observation_shape, tuple)
    tf.compat.v1.logging.info('Creating %s agent with the following parameters:',
                    self.__class__.__name__)
    tf.compat.v1.logging.info('\t num_sub_actions: %s', str(num_sub_actions))
    tf.compat.v1.logging.info('\t network: %s', str(network))
    tf.compat.v1.logging.info('\t mixing_network: %s', str(mixing_network))
    tf.compat.v1.logging.info('\t use_dueling: %s', use_dueling)
    tf.compat.v1.logging.info('\t double_dqn: %s', double_dqn)
    tf.compat.v1.logging.info('\t gamma: %f', gamma)
    tf.compat.v1.logging.info('\t update_horizon: %f', update_horizon)
    tf.compat.v1.logging.info('\t min_replay_history: %d', min_replay_history)
    tf.compat.v1.logging.info('\t update_period: %d', update_period)
    tf.compat.v1.logging.info('\t target_update_period: %d', target_update_period)
    tf.compat.v1.logging.info('\t epsilon_train: %f', epsilon_train)
    tf.compat.v1.logging.info('\t epsilon_eval: %f', epsilon_eval)
    tf.compat.v1.logging.info('\t epsilon_decay_period: %d', epsilon_decay_period)
    tf.compat.v1.logging.info('\t replay_scheme: %s', replay_scheme)
    tf.compat.v1.logging.info('\t tf_device: %s', tf_device)
    tf.compat.v1.logging.info('\t loss_type: %s', loss_type)
    tf.compat.v1.logging.info('\t optimizer: %s', optimizer)

    self.num_sub_actions = num_sub_actions
    self.observation_shape = tuple(observation_shape)
    self.observation_dtype = observation_dtype
    self.stack_size = stack_size
    self.network = network
    self.mixing_network = mixing_network
    self.use_dueling = use_dueling
    self._double_dqn = double_dqn
    self.gamma = gamma
    self.update_horizon = update_horizon
    self.cumulative_gamma = math.pow(gamma, update_horizon)
    self.min_replay_history = min_replay_history
    self.update_period = update_period
    self.target_update_period = target_update_period
    self.epsilon_fn = epsilon_fn
    self.epsilon_train = epsilon_train
    self.epsilon_eval = epsilon_eval
    self.epsilon_decay_period = epsilon_decay_period
    self._replay_scheme = replay_scheme
    self.eval_mode = False
    self.training_steps = 0
    self.loss_type = loss_type
    self.optimizer = optimizer
    self.summary_writer = summary_writer
    self.summary_writing_frequency = summary_writing_frequency

    if double_dqn: raise NotImplementedError()
    if replay_scheme == 'prioritized': raise NotImplementedError()

    with tf.device(tf_device):
      state_shape = (1,) + self.observation_shape + (stack_size,)
      self.state = np.zeros(state_shape)
      self.state_ph = tf.compat.v1.placeholder(self.observation_dtype, state_shape,
                                     name='state_ph')
      self._replay = self._build_replay_buffer(use_staging)
      self._build_networks()
      self._train_op = self._build_train_op()
      self._sync_qt_ops = self._build_sync_op()

    if self.summary_writer is not None:
      self._merged_summaries = tf.compat.v1.Summary.merge_all()
    self._sess = sess
    self._saver = tf.compat.v1.train.Saver(max_to_keep=max_tf_checkpoints_to_keep)
    self._observation = None
    self._last_observation = None

  def _get_mixing_network_type(self):
    return collections.namedtuple('MIX_network', ['q_value'])

  def _mixing_network_template(self, utils, v_value):
    kwargs = {}
    kwargs['use_dueling'] = self.use_dueling
    kwargs['v_value'] = v_value
    return self.mixing_network(self._get_mixing_network_type(), utils,
        **kwargs)

  def _build_networks(self):
    self.online_convnet = tf.compat.v1.make_template('Online', self._network_template)
    self.target_convnet = tf.compat.v1.make_template('Target', self._network_template)
    self.online_mixnet = tf.compat.v1.make_template('Online/Mix', self._mixing_network_template)
    self.target_mixnet = tf.compat.v1.make_template('Target/Mix', self._mixing_network_template)
    self._net_outputs = self.online_convnet(self.state_ph)

    self._bq_argmax = []
    for bq_values_dim in self._net_outputs.bq_values:
      self._bq_argmax.append(tf.argmax(bq_values_dim, axis=1)[0])

    self._replay_net_outputs = self.online_convnet(self._replay.states)
    self._replay_next_target_net_outputs = self.target_convnet(
        self._replay.next_states)

    replay_chosen_bq = []
    replay_sub_actions = tf.unstack(self._replay.actions, axis=1)
    for dim, (num_sub_actions_dim, replay_sub_actions_dim,
        replay_bq_values_dim) in enumerate(zip(
        self.num_sub_actions, replay_sub_actions,
        self._replay_net_outputs.bq_values)):
      replay_sub_actions_one_hot_dim = tf.one_hot(
          replay_sub_actions_dim, num_sub_actions_dim, 1., 0.,
          name='sub_action_one_hot_{}'.format(dim))
      replay_chosen_bq_dim = tf.reduce_sum(
          replay_bq_values_dim * replay_sub_actions_one_hot_dim,
          reduction_indices=1, name='replay_chosen_bq_{}'.format(dim))
      replay_chosen_bq.append(replay_chosen_bq_dim)

    replay_next_bqt_max = []
    for next_bqt_values_dim in self._replay_next_target_net_outputs.bq_values:
      replay_next_bqt_max.append(tf.reduce_max(next_bqt_values_dim, 1))
    
    self.replay_chosen_q = self.online_mixnet(replay_chosen_bq,
        self._replay_net_outputs.v_value)
    self.replay_next_qt_max = self.target_mixnet(replay_next_bqt_max,
        self._replay_next_target_net_outputs.v_value)

  def _build_target_q_op(self):
    self.replay_next_qt_max = tf.squeeze(self.replay_next_qt_max.q_value)
    return self._replay.rewards + self.cumulative_gamma * self.replay_next_qt_max * (
        1. - tf.cast(self._replay.terminals, tf.float32))

  def _build_train_op(self):
    self.replay_chosen_q = tf.squeeze(self.replay_chosen_q.q_value)
    target = tf.stop_gradient(self._build_target_q_op())
    if self.loss_type == 'Huber':
      loss = tf.losses.huber_loss(
          target, self.replay_chosen_q, reduction=tf.compat.v1.losses.Reduction.NONE)
    elif self.loss_type == 'MSE':
      loss = tf.compat.v1.losses.mean_squared_error(
          target, self.replay_chosen_q, reduction=tf.compat.v1.losses.Reduction.NONE)
    else:
      raise ValueError('Invalid loss type: {}'.format(self.loss_type))

    if self.summary_writer is not None:
      with tf.variable_scope('Losses'):
        tf.compat.v1.Summary.scalar(self.loss_type+'Loss', tf.reduce_mean(loss))
    return self.optimizer.minimize(tf.reduce_mean(loss))
