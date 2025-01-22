import os, sys
# Get the absolute path of the parent directory of the current notebook
notebook_dir = os.path.abspath('')
project_root = os.path.dirname(notebook_dir)  # Assumes the notebook is in root/notebooks

# Add the project root to the sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import random
import requests
import os
from src.utils.config_manager import Config
from initialize import initialize_environment, initialize_agent, match_random_sound, format_match_json_from_state

# URL of the Flask app
BASE_URL = "http://127.0.0.1:5000"

# List of knob IDs
knob_ids = ["Cutoff-Frequency", "Attack", "Decay", "Sustain", "Release"]

def update_knobs_periodically():
    while True:
        for knob in knob_ids:
            value = random.randint(0, 100)
            response = requests.post(f"{BASE_URL}/update-knob", json={"id": knob, "value": value})
            if response.status_code == 200:
                print(f"Knob {knob} randomly set to {value}")
            else:
                print(f"Failed to update {knob}: {response.text}")
        time.sleep(5)

def match_random_and_send_periodically(env, agent, mapping_file, sleep=10):
    api_url = f"{BASE_URL}/send-match"
    while True:
        env = match_random_sound(env, agent)
        match_data = format_match_json_from_state(env, mapping_file)

        print(f"Calling URL: {api_url}, with data:")
        print(match_data)
        response = requests.post(api_url, json=match_data)
        print(f"Response: {response}")
        print('################################ \n')

        time.sleep(sleep)


if __name__ == "__main__":
    # update_knobs_periodically() # Disabled for now

    # Load config
    script_dir = os.getcwd()
    config = Config()
    config_path = os.path.join(script_dir, "configs", "train_end_to_end.yaml")
    config.load(config_path)

    # Initialize setup
    print("Initializing environment...")

    env = initialize_environment(config)
    agent = initialize_agent(env, config)
    script_dir = os.getcwd()
    mapping_path = os.path.join(script_dir, "configs", "synth_parameter_mapping.yaml")

    # Run periodic random match and send
    print("Starting period random match loop")
    match_random_and_send_periodically(env, agent, mapping_file=mapping_path, sleep=3)