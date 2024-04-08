import tensorflow as tf
from tensorflow.keras import layers, models, optimizers


class ActorCriticAgent(tf.keras.Model):
    def __init__(self, observer_network, action_dim, hidden_dim=256, gamma=0.99):
        super(ActorCriticAgent, self).__init__()
        self.observer_network = observer_network
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma

        # Ensure observer network's weights are trainable within the ActorCriticAgent
        self.observer_network.trainable = True

        # Actor network
        self.actor_dense1 = layers.Dense(hidden_dim, activation='relu')
        self.actor_dense2 = layers.Dense(hidden_dim, activation='relu')
        self.actor_output = layers.Dense(action_dim, activation='tanh')  # For continuous actions

        # Critic network
        self.critic_dense1 = layers.Dense(hidden_dim, activation='relu')
        self.critic_dense2 = layers.Dense(hidden_dim, activation='relu')
        self.critic_output = layers.Dense(1, activation='linear')

        # Optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = optimizers.Adam(learning_rate=1e-3)

    def call(self, inputs, training=False, action=None):
        features = self.observer_network(inputs, training=training)

        # Actor
        x_actor = self.actor_dense1(features)
        x_actor = self.actor_dense2(x_actor)
        actor_output = self.actor_output(x_actor)

        # Critic
        # Use provided action if available, otherwise use the actor's output
        action_to_use = action if action is not None else actor_output
        critic_input = tf.concat([features, action_to_use], axis=-1)
        x_critic = self.critic_dense1(critic_input)
        x_critic = self.critic_dense2(x_critic)
        critic_output = self.critic_output(x_critic)

        return actor_output, critic_output

    def train_step(self, data):
        states, actions, rewards, next_states, dones = data

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Predict actions for the next states
            next_actions, _ = self(next_states, training=True)

            # Get critic values for next state-action pairs
            _, next_critic_values = self(next_states, training=True, action=next_actions)
            next_critic_values = tf.squeeze(next_critic_values, axis=1)

            # Get critic values for the current state-action pairs
            _, critic_values = self(states, training=True, action=actions)
            critic_values = tf.squeeze(critic_values, axis=1)

            # Critic loss
            target_values = rewards + (1 - dones) * self.gamma * next_critic_values
            critic_loss = tf.keras.losses.MSE(target_values, critic_values)

            # Actor loss (use critic to evaluate the actions predicted by the actor)
            predicted_actions, _ = self(states, training=True)
            _, critic_values_for_actor_loss = self(states, training=True, action=predicted_actions)
            critic_values_for_actor_loss = tf.squeeze(critic_values_for_actor_loss, axis=1)
            actor_loss = -tf.reduce_mean(critic_values_for_actor_loss)

        # Compute gradients and apply updates
        critic_grad = tape.gradient(critic_loss,
                                    self.critic_dense1.trainable_variables + self.critic_dense2.trainable_variables + self.critic_output.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad,
                                                  self.critic_dense1.trainable_variables + self.critic_dense2.trainable_variables + self.critic_output.trainable_variables))

        actor_grad = tape.gradient(actor_loss,
                                   self.actor_dense1.trainable_variables + self.actor_dense2.trainable_variables + self.actor_output.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad,
                                                 self.actor_dense1.trainable_variables + self.actor_dense2.trainable_variables + self.actor_output.trainable_variables))

        del tape  # Free tape memory

        return {"actor_loss": actor_loss.numpy(), "critic_loss": critic_loss.numpy()}

    def act(self, state):
        state = tf.expand_dims(state, 0)
        action, _ = self(state, training=False)
        return action[0].numpy()
