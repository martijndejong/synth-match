import time
import random
import requests

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

if __name__ == "__main__":
    update_knobs_periodically()
