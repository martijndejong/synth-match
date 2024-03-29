"""
TODO: Design actual 'learning loop' with agent and environment interaction -- current code is placeholder
"""

from src.agents.actor_critic_agent import ActorCriticAgent
from src.utils.observers import SpectrogramObserver
from src.environments.synth_a_env import SynthA

# Instantiate the model with observers
agent = ActorCriticAgent(observer=SpectrogramObserver())

# Compile the model
agent.compile(optimizer='adam', loss='your_loss_function', metrics=['accuracy'])

# Super rough outline of RL training loop below, will need complete revision
env = SynthA()
new_state = env.reset()

done = False
while not done:
    action = agent.call(new_state)

    new_state, reward, done = env.step(action)
