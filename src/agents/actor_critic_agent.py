import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from src.environment.environment import State  # Import the State dataclass


class ActorCriticAgent(tf.keras.Model):
    def __init__(self, observer_network, action_dim, hidden_dim=256, gamma=0.99):
        super(ActorCriticAgent, self).__init__()
        self.observer_network = observer_network
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # Ensure observer network's weights are trainable within the ActorCriticAgent
        self.observer_network.trainable = True

        # Actor sub-model
        self.actor = models.Sequential([
            layers.InputLayer(input_shape=(128 + action_dim,)),  # TODO: hardcoded 128 must be aligned with observer
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(action_dim, activation='tanh')  # For continuous actions
        ], name='actor')

        # Critic sub-model
        self.critic = models.Sequential([
            layers.InputLayer(input_shape=(128 + action_dim * 2,)),  # TODO: hardcoded 128 must be aligned with observer
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(1, activation='linear')
        ], name='critic')

        # Optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs, synth_params, training=False, action=None):
        features = self.observer_network(inputs, training=training)
        # Concatenate the features and synth_params
        concat_input = tf.concat([features, synth_params], axis=-1)

        # Actor forward pass
        actor_output = self.actor(concat_input)

        # Critic
        # Use provided action if available, otherwise use the actor's output
        action_to_use = action if action is not None else actor_output
        critic_input = tf.concat([concat_input, action_to_use], axis=-1)

        # Critic forward pass
        critic_output = self.critic(critic_input)

        return actor_output, critic_output

    def train_step(self, data):
        # States: What our environment returns, i.e., spectrogram
        # Synth params: Known input synth parameters
        states, synth_params, actions, rewards, next_states, next_synth_params, dones = data

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        synth_params = tf.convert_to_tensor(synth_params, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        next_synth_params = tf.convert_to_tensor(next_synth_params, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Predict actions for the next states
            next_actions, _ = self(next_states, next_synth_params, training=True)

            # Get critic values for next state-action pairs
            _, next_critic_values = self(next_states, next_synth_params, training=True, action=next_actions)
            next_critic_values = tf.squeeze(next_critic_values, axis=1)

            # Get critic values for the current state-action pairs
            _, critic_values = self(states, synth_params, training=True, action=actions)
            critic_values = tf.squeeze(critic_values, axis=1)

            # Critic loss
            target_values = rewards + (1 - dones) * self.gamma * next_critic_values
            critic_loss = tf.keras.losses.MSE(target_values, critic_values)

            # Actor loss (use critic to evaluate the actions predicted by the actor)
            predicted_actions, _ = self(states, synth_params, training=True)
            _, critic_values_for_actor_loss = self(states, synth_params, training=True, action=predicted_actions)
            critic_values_for_actor_loss = tf.squeeze(critic_values_for_actor_loss, axis=1)
            actor_loss = -tf.reduce_mean(critic_values_for_actor_loss)

        # Compute gradients for both actor and critic including the observer network
        critic_grad = tape.gradient(critic_loss,
                                    self.critic.trainable_variables + self.observer_network.trainable_variables)
        actor_grad = tape.gradient(actor_loss,
                                   self.actor.trainable_variables + self.observer_network.trainable_variables)

        # Apply gradients
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables + self.observer_network.trainable_variables))
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables + self.observer_network.trainable_variables))

        del tape  # Free tape memory

        return {"actor_loss": actor_loss.numpy(), "critic_loss": critic_loss.numpy()}

    def act(self, state, synth_params):
        state = tf.expand_dims(state, 0)
        synth_params = tf.expand_dims(synth_params, 0)
        action, _ = self(state, synth_params, training=False)
        return action[0].numpy()

    # Methods to save and load network for pretraining
    def save_actor_critic_weights(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_actor_critic_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
