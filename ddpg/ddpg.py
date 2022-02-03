import tensorflow as tf
import tensorflow.keras as keras
import asyncio
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import PolicyNetwork, QfunNetwork


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.001, env=None,
                 gamma=0.99, n_actions=2, buffer_size=10000, tau=0.005,
                 qfun_layers=[40, 30], policy_layers=[40, 30], batch_size=100,
                 noise=0.1):

        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.memory = ReplayBuffer(buffer_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high
        self.min_action = env.action_space.low

        self.policy = PolicyNetwork(layer_sizes=policy_layers, 
                                    n_actions=n_actions, name='policy')
        self.qfunction = QfunNetwork(layer_sizes=qfun_layers, name='qfunction')
        self.target_policy = PolicyNetwork(layer_sizes=policy_layers,
                                     n_actions=n_actions, name='target_policy')
        self.target_qfunction = QfunNetwork(layer_sizes=qfun_layers,
                                            name='target_qfunction')

        self.policy.compile(optimizer=Adam(learning_rate=alpha))
        self.qfunction.compile(optimizer=Adam(learning_rate=beta))
        self.target_policy.compile(optimizer=Adam(learning_rate=alpha))
        self.target_qfunction.compile(optimizer=Adam(learning_rate=beta))

        self.update_target_parameters(tau=1)

    def update_target_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_policy.weights
        for i, weight in enumerate(self.policy.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_policy.set_weights(weights)

        weights = []
        targets = self.target_qfunction.weights
        for i, weight in enumerate(self.qfunction.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_qfunction.set_weights(weights)

    def remember(self, state, action, cost, new_state, done):
        self.memory.store_transition(state, action, cost, new_state, done)

    def save_models(self):
        print('... saving models ...')
        try:
            self.policy.save_weights(self.policy.checkpoint_file)
            self.target_policy.save_weights(self.target_policy.checkpoint_file)
            self.qfunction.save_weights(self.qfunction.checkpoint_file)
            self.target_qfunction.save_weights(self.target_qfunction.checkpoint_file)
        except Exception:
            print('failed to save weights')

    def load_models(self):
        print('... loading models ...')
        self.policy.load_weights(self.policy.checkpoint_file).expect_partial()
        self.target_policy.load_weights(
                         self.target_policy.checkpoint_file).expect_partial()
        self.qfunction.load_weights(
                             self.qfunction.checkpoint_file).expect_partial()
        self.target_qfunction.load_weights(
                        self.target_qfunction.checkpoint_file).expect_partial()

    def choose_action(self, observation, action_noise=True):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.policy(state)
        if action_noise:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, cost, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        costs = tf.convert_to_tensor(cost, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_policy(states_)
            qfunction_value_ = tf.squeeze(self.target_qfunction(
                                states_, target_actions), 1)
            qfunction_value = tf.squeeze(self.qfunction(states, actions), 1)
            target = costs + self.gamma*qfunction_value_*(1-done)
            qfunction_loss = keras.losses.MSE(target, qfunction_value)

        qfunction_network_gradient = tape.gradient(qfunction_loss,
                                            self.qfunction.trainable_variables)
        self.qfunction.optimizer.apply_gradients(zip(
            qfunction_network_gradient, self.qfunction.trainable_variables))

        with tf.GradientTape() as tape:
            new_policy_actions = self.policy(states)
            policy_loss = self.qfunction(states, new_policy_actions)
            policy_loss = tf.math.reduce_mean(policy_loss)

        policy_network_gradient = tape.gradient(policy_loss,
                                               self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(zip(
            policy_network_gradient, self.policy.trainable_variables))

        self.update_target_parameters()