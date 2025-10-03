import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import utils


class ModelBasedPolicy(object):

    def __init__(self,
                 env,
                 init_dataset,
                 horizon=15,
                 num_random_action_selection=4096,
                 nn_layers=1):
        self._cost_fn = env.cost_fn
        self._state_dim = env.observation_space.shape[0]
        self._action_dim = env.action_space.shape[0]
        self._action_space_low = env.action_space.low
        self._action_space_high = env.action_space.high
        self._dataset = init_dataset
        self._horizon = horizon
        self._num_random_action_selection = num_random_action_selection
        self._nn_layers = nn_layers
        self._learning_rate = 1e-3

        (
            self._sess,
            self._state_ph,
            self._action_ph,
            self._next_state_ph,
            self._next_state_pred,
            self._loss,
            self._optimizer,
            self._action_state_ph,
            self._best_action,
        ) = self._setup_graph()

    def _setup_placeholders(self):
        """
            Creates the placeholders used for training, prediction, and action selection

            returns:
                state_ph: current state
                action_ph: current_action
                next_state_ph: next state

            implementation details:
                (a) the placeholders should have 2 dimensions,
                    in which the 1st dimension is variable length (i.e., None)
        """
        state_ph = tf.placeholder(tf.float32, shape=[None, self._state_dim], name='state')
        action_ph = tf.placeholder(tf.float32, shape=[None, self._action_dim], name='action')
        next_state_ph = tf.placeholder(tf.float32, shape=[None, self._state_dim], name='next_state')

        return state_ph, action_ph, next_state_ph

    def _dynamics_func(self, state, action, reuse):
        """
            Takes as input a state and action, and predicts the next state

            returns:
                next_state_pred: predicted next state

            implementation details (in order):
                (a) Normalize both the state and action by using the statistics of self._init_dataset and
                    the utils.normalize function
                (b) Concatenate the normalized state and action
                (c) Pass the concatenated, normalized state-action tensor through a neural network with
                    self._nn_layers number of layers using the function utils.build_mlp. The resulting output
                    is the normalized predicted difference between the next state and the current state
                (d) Unnormalize the delta state prediction, and add it to the current state in order to produce
                    the predicted next state

        """
        dataset = self._dataset
        state_mean = dataset.state_mean
        state_std = dataset.state_std
        action_mean = dataset.action_mean
        action_std = dataset.action_std
        delta_mean = dataset.delta_state_mean
        delta_std = dataset.delta_state_std

        state_norm = utils.normalize(state, state_mean, state_std)
        action_norm = utils.normalize(action, action_mean, action_std)
        inputs = tf.concat([state_norm, action_norm], axis=1)

        delta_norm_pred = utils.build_mlp(
            inputs,
            self._state_dim,
            scope='dynamics',
            n_layers=self._nn_layers,
            hidden_dim=500,
            activation=tf.nn.relu,
            output_activation=None,
            reuse=reuse,
        )

        delta_pred = utils.unnormalize(delta_norm_pred, delta_mean, delta_std)
        next_state_pred = state + delta_pred

        return next_state_pred

    def _setup_training(self, state_ph, next_state_ph, next_state_pred):
        """
            Takes as input the current state, next state, and predicted next state, and returns
            the loss and optimizer for training the dynamics model

            returns:
                loss: Scalar loss tensor
                optimizer: Operation used to perform gradient descent

            implementation details (in order):
                (a) Compute both the actual state difference and the predicted state difference
                (b) Normalize both of these state differences by using the statistics of self._init_dataset and
                    the utils.normalize function
                (c) The loss function is the mean-squared-error between the normalized state difference and
                    normalized predicted state difference
                (d) Create the optimizer by minimizing the loss using the Adam optimizer with self._learning_rate

        """
        dataset = self._dataset
        delta_true = next_state_ph - state_ph
        delta_pred = next_state_pred - state_ph

        delta_true_norm = utils.normalize(delta_true, dataset.delta_state_mean, dataset.delta_state_std)
        delta_pred_norm = utils.normalize(delta_pred, dataset.delta_state_mean, dataset.delta_state_std)

        loss = tf.reduce_mean(tf.square(delta_true_norm - delta_pred_norm))
        optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(loss)

        return loss, optimizer

    def _setup_action_selection(self, state_ph):
        """
            Computes the best action from the current state by using randomly sampled action sequences
            to predict future states, evaluating these predictions according to a cost function,
            selecting the action sequence with the lowest cost, and returning the first action in that sequence

            returns:
                best_action: the action that minimizes the cost function (tensor with shape [self._action_dim])

            implementation details (in order):
                (a) We will assume state_ph has a batch size of 1 whenever action selection is performed
                (b) Randomly sample uniformly self._num_random_action_selection number of action sequences,
                    each of length self._horizon
                (c) Starting from the input state, unroll each action sequence using your neural network
                    dynamics model
                (d) While unrolling the action sequences, keep track of the cost of each action sequence
                    using self._cost_fn
                (e) Find the action sequence with the lowest cost, and return the first action in that sequence

            Hints:
                (i) self._cost_fn takes three arguments: states, actions, and next states. These arguments are
                    2-dimensional tensors, where the 1st dimension is the batch size and the 2nd dimension is the
                    state or action size
                (ii) You should call self._dynamics_func and self._cost_fn a total of self._horizon times
                (iii) Use tf.random_uniform(...) to generate the random action sequences

        """
        action_low = tf.constant(self._action_space_low, dtype=tf.float32)
        action_high = tf.constant(self._action_space_high, dtype=tf.float32)

        random_actions = tf.random_uniform(
            shape=[self._num_random_action_selection, self._horizon, self._action_dim],
            minval=0.0,
            maxval=1.0,
            dtype=tf.float32,
        )
        action_range = action_high - action_low
        action_range = tf.reshape(action_range, [1, 1, self._action_dim])
        action_low_reshaped = tf.reshape(action_low, [1, 1, self._action_dim])
        sampled_actions = action_low_reshaped + random_actions * action_range

        tiled_states = tf.tile(state_ph, [self._num_random_action_selection, 1])
        total_cost = tf.zeros([self._num_random_action_selection], dtype=tf.float32)

        current_states = tiled_states
        for t in range(self._horizon):
            current_actions = sampled_actions[:, t, :]
            next_states = self._dynamics_func(current_states, current_actions, reuse=True)
            step_cost = self._cost_fn(current_states, current_actions, next_states)
            if tf.is_tensor(step_cost) and step_cost.shape.ndims is not None and step_cost.shape.ndims > 1:
                step_cost = tf.reduce_sum(step_cost, axis=1)
            total_cost += step_cost
            current_states = next_states

        best_indices = tf.argmin(total_cost, axis=0)
        first_actions = sampled_actions[:, 0, :]
        best_action = tf.gather(first_actions, best_indices)

        return best_action

    def _setup_graph(self):
        """
        Sets up the tensorflow computation graph for training, prediction, and action selection

        The variables returned will be set as class attributes (see __init__)
        """
        graph = tf.Graph()
        with graph.as_default():
            state_ph, action_ph, next_state_ph = self._setup_placeholders()
            next_state_pred = self._dynamics_func(state_ph, action_ph, reuse=False)
            loss, optimizer = self._setup_training(state_ph, next_state_ph, next_state_pred)

            action_state_ph = tf.placeholder(tf.float32, shape=[1, self._state_dim], name='action_state')
            best_action = self._setup_action_selection(action_state_ph)

            init = tf.global_variables_initializer()

        sess = tf.Session(graph=graph)
        sess.run(init)

        return (
            sess,
            state_ph,
            action_ph,
            next_state_ph,
            next_state_pred,
            loss,
            optimizer,
            action_state_ph,
            best_action,
        )

    def train_step(self, states, actions, next_states):
        """
        Performs one step of gradient descent

        returns:
            loss: the loss from performing gradient descent
        """
        feed_dict = {
            self._state_ph: states,
            self._action_ph: actions,
            self._next_state_ph: next_states,
        }
        _, loss_val = self._sess.run([self._optimizer, self._loss], feed_dict=feed_dict)
        return loss_val

    def predict(self, state, action):
        """
        Predicts the next state given the current state and action

        returns:
            next_state_pred: predicted next state

        implementation detils:
            (i) The state and action arguments are 1-dimensional vectors (NO batch dimension)
        """
        assert np.shape(state) == (self._state_dim,)
        assert np.shape(action) == (self._action_dim,)
        feed_dict = {
            self._state_ph: state[None, :],
            self._action_ph: action[None, :],
        }
        next_state_pred = self._sess.run(self._next_state_pred, feed_dict=feed_dict)
        next_state_pred = next_state_pred[0]

        assert np.shape(next_state_pred) == (self._state_dim,)
        return next_state_pred

    def get_action(self, state):
        """
        Computes the action that minimizes the cost function given the current state

        returns:
            best_action: the best action
        """
        assert np.shape(state) == (self._state_dim,)
        feed_dict = {
            self._action_state_ph: state[None, :],
        }
        best_action = self._sess.run(self._best_action, feed_dict=feed_dict)

        assert np.shape(best_action) == (self._action_dim,)
        return best_action

    def update_dataset(self, dataset):
        self._dataset = dataset