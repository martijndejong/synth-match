import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np


class TD3Agent(tf.keras.Model):
    def __init__(
            self,
            observer_network,
            action_dim,
            hidden_dim=256,
            gamma=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
    ):
        super(TD3Agent, self).__init__()
        self.observer_network = observer_network
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay

        # Training step counter
        self.total_it = 0

        # Ensure observer network's weights are trainable within the TD3Agent
        self.observer_network.trainable = True

        # Actor network
        self.actor = self.build_actor()
        self.actor_target = self.build_actor()
        self.actor_target.set_weights(self.actor.get_weights())

        # Critic networks
        self.critic_1 = self.build_critic()
        self.critic_2 = self.build_critic()
        self.critic_1_target = self.build_critic()
        self.critic_2_target = self.build_critic()
        self.critic_1_target.set_weights(self.critic_1.get_weights())
        self.critic_2_target.set_weights(self.critic_2.get_weights())

        # Optimizers
        self.actor_optimizer = optimizers.Adam(learning_rate=1e-4)
        self.critic_optimizer = optimizers.Adam(learning_rate=1e-3)
        self.observer_optimizer = optimizers.Adam(learning_rate=1e-5)  # Super slow learning for pre-trained observer

    def build_actor(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=(128 + self.action_dim,)),  # ALIGN SIZE WITH OBSERVER OUTPUT
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(self.action_dim, activation='tanh'),  # For incremental continuous actions
            # layers.Dense(self.action_dim, activation='sigmoid'),  # For absolute control mode
            # layers.Lambda(lambda x: x * 0.1)  # Apply scaling factor to tanh output
        ])
        return model

    def build_critic(self):
        model = models.Sequential([
            layers.InputLayer(input_shape=(128 + self.action_dim * 2,)),  # ALIGN SIZE WITH OBSERVER OUTPUT
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(self.hidden_dim, activation='relu'),
            layers.Dense(1, activation='linear')
        ])
        return model

    def call(self, inputs, synth_params, training=False, action=None):
        # This method is not typically used in TD3, but we'll implement it to maintain compatibility
        features = self.observer_network(inputs, training=training)
        concat_input = tf.concat([features, synth_params], axis=-1)

        # Actor forward pass
        actor_output = self.actor(concat_input)

        # Critic forward pass
        if action is None:
            action = actor_output
        critic_input = tf.concat([concat_input, action], axis=-1)
        q1 = self.critic_1(critic_input)
        q2 = self.critic_2(critic_input)

        # Return actor output and minimum of the two critic outputs
        return actor_output, tf.minimum(q1, q2)

    def act(self, state, synth_params):
        state = tf.expand_dims(state, 0)
        synth_params = tf.expand_dims(synth_params, 0)
        features = self.observer_network(state)
        concat_input = tf.concat([features, synth_params], axis=-1)
        action = self.actor(concat_input)
        return action[0].numpy()

    def train_step(self, data):
        self.total_it += 1

        states, synth_params, actions, rewards, next_states, next_synth_params, dones = data

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        synth_params = tf.convert_to_tensor(synth_params, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        next_synth_params = tf.convert_to_tensor(next_synth_params, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Update Critic networks
        with tf.GradientTape(persistent=True) as tape:
            # Get features for current and next states
            features = self.observer_network(states, training=True)
            next_features = self.observer_network(next_states, training=True)

            # Concatenate features with synth_params
            concat_input = tf.concat([features, synth_params], axis=-1)
            next_concat_input = tf.concat([next_features, next_synth_params], axis=-1)

            # Target policy smoothing
            noise = tf.clip_by_value(
                tf.random.normal(shape=actions.shape, stddev=self.policy_noise),
                -self.noise_clip, self.noise_clip)

            # Compute target actions
            next_actions = self.actor_target(next_concat_input)
            next_actions = tf.clip_by_value(next_actions + noise, -1.0, 1.0)

            # Compute target Q-values
            target_critic_input = tf.concat([next_concat_input, next_actions], axis=-1)
            target_q1 = self.critic_1_target(target_critic_input)
            target_q2 = self.critic_2_target(target_critic_input)
            target_q = tf.minimum(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * tf.squeeze(target_q, axis=1)

            # Current Q-values
            critic_input = tf.concat([concat_input, actions], axis=-1)
            current_q1 = tf.squeeze(self.critic_1(critic_input), axis=1)
            current_q2 = tf.squeeze(self.critic_2(critic_input), axis=1)

            # Critic losses
            critic_loss1 = tf.keras.losses.MSE(target_q, current_q1)
            critic_loss2 = tf.keras.losses.MSE(target_q, current_q2)
            critic_loss = critic_loss1 + critic_loss2

        # Compute and apply gradients for critics
        # Observer network is updated in critic learning step
        critic_vars = self.critic_1.trainable_variables + self.critic_2.trainable_variables
        observer_vars = self.observer_network.trainable_variables
        # Calculate gradient for critic and observer, then split and apply separately
        critic_grads_and_obs = tape.gradient(critic_loss, critic_vars + observer_vars)
        critic_grads = critic_grads_and_obs[:len(critic_vars)]
        observer_grads = critic_grads_and_obs[len(critic_vars):]

        self.critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))
        self.observer_optimizer.apply_gradients(zip(observer_grads, observer_vars))

        # Delayed policy updates
        actor_loss = 0.0  # Initialize actor_loss
        if self.total_it % self.policy_delay == 0:
            with tf.GradientTape() as tape_actor:
                # Compute actor loss
                actor_actions = self.actor(concat_input)
                actor_critic_input = tf.concat([concat_input, actor_actions], axis=-1)
                actor_q1 = self.critic_1(actor_critic_input)
                actor_loss = -tf.reduce_mean(actor_q1)

            # Compute and apply gradients for actor
            actor_vars = self.actor.trainable_variables
            actor_grads = tape_actor.gradient(actor_loss, actor_vars)
            self.actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

            # Soft update of target networks
            self.update_target_weights(self.actor_target.variables, self.actor.variables)
            self.update_target_weights(self.critic_1_target.variables, self.critic_1.variables)
            self.update_target_weights(self.critic_2_target.variables, self.critic_2.variables)

        del tape  # Free tape memory

        return {
            "actor_loss": actor_loss if isinstance(actor_loss, tf.Tensor) else tf.constant(actor_loss,
                                                                                           dtype=tf.float32),
            "critic_loss": critic_loss.numpy()
        }

    def update_target_weights(self, target_weights, source_weights):
        for (target, source) in zip(target_weights, source_weights):
            target.assign(self.tau * source + (1 - self.tau) * target)

    # Methods to save and load network for pretraining
    def save_actor_critic_weights(self, actor_path, critic_path):
        self.actor.save_weights(actor_path)
        # Save both critic networks' weights together
        self.critic_1.save_weights(critic_path + '_1.h5')
        self.critic_2.save_weights(critic_path + '_2.h5')

    def load_actor_critic_weights(self, actor_path, critic_path):
        self.actor.load_weights(actor_path)
        self.actor_target.load_weights(actor_path)
        self.critic_1.load_weights(critic_path + '_1.h5')
        self.critic_2.load_weights(critic_path + '_2.h5')
        self.critic_1_target.load_weights(critic_path + '_1.h5')
        self.critic_2_target.load_weights(critic_path + '_2.h5')
