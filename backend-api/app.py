from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

# Suppress Flask default access logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Mock data to store knob states
knob_states = {
    "Cutoff-Frequency": 50,
    "Attack": 50,
    "Decay": 50,
    "Sustain": 50,
    "Release": 50,
}

# Endpoint to update the knob value (POST)
@app.route('/update-knob', methods=['POST'])
def update_knob():
    data = request.json
    knob_id = data.get('id')
    value = data.get('value')

    if knob_id in knob_states:
        knob_states[knob_id] = value  # Update state
        print(f"Knob {knob_id} is changed to {value}")
        # Notify all clients about the updated knob state
        socketio.emit('knob_update', {"id": knob_id, "value": value})
        return jsonify({"message": f"Knob {knob_id} updated to {value}"}), 200
    else:
        return jsonify({"error": "Knob ID not found"}), 404

# Endpoint to fetch the knob value (GET)
@app.route('/get-knob', methods=['GET'])
def get_knob():
    knob_id = request.args.get('id')  # Retrieve knob ID from query params

    if knob_id in knob_states:
        return jsonify({"id": knob_id, "value": knob_states[knob_id]}), 200
    else:
        return jsonify({"error": "Knob ID not found"}), 404

if __name__ == '__main__':
    socketio.run(app, debug=True)
