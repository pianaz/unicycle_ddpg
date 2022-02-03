import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

class QfunNetwork(keras.Model):
    def __init__(self, layer_sizes= [400, 300], name='Qfunction',
                 chkpt_dir='tmp/ddpg'):
        super(QfunNetwork, self).__init__()

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                                            self.model_name+'_ddpg')

        self.hidden_layers = []
        self.layers_num = len(layer_sizes)
        for i in range(self.layers_num):
            self.hidden_layers.append(Dense(layer_sizes[i], activation='relu'))
        self.q = Dense(1, activation=None)

    def call(self, state, action):
        action_value = self.hidden_layers[0](tf.concat([state,action], axis=1))
        for i in range(1, self.layers_num):
            action_value = self.hidden_layers[i](action_value)
        q = self.q(action_value)

        return q

class PolicyNetwork(keras.Model):
    def __init__(self, layer_sizes= [400, 300], n_actions=2, name='policy',
            chkpt_dir='tmp/ddpg'):
        super(PolicyNetwork, self).__init__()
        self.n_actions = n_actions

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 
                                            self.model_name+'_ddpg')

        self.hidden_layers = []
        self.layers_num = len(layer_sizes)
        for i in range(self.layers_num):
            self.hidden_layers.append(Dense(layer_sizes[i], activation='relu'))
        self.mu = Dense(self.n_actions, activation= 'tanh') 

    def call(self, state):
        tmp = self.hidden_layers[0](state)
        for i in range(1, self.layers_num):
            tmp = self.hidden_layers[i](tmp)
        mu = self.mu(tmp)

        return mu
