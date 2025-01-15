from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import logging

# Suppress Flask default access logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Initialize app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Mock knob states
knob_states = {
    "Cutoff-Frequency": 50,
    "Attack": 50,
    "Decay": 50,
    "Sustain": 50,
    "Release": 50,
}

@app.route('/update-knob', methods=['POST'])
def update_knob():
    data = request.json
    knob_id = data.get('id')
    value = data.get('value')
    if knob_id in knob_states:
        knob_states[knob_id] = value
        socketio.emit('knob_update', {"id": knob_id, "value": value})
        return jsonify({"message": f"Knob {knob_id} updated to {value}"}), 200
    return jsonify({"error": "Knob ID not found"}), 404

@app.route('/get-knob', methods=['GET'])
def get_knob():
    knob_id = request.args.get('id')
    if knob_id in knob_states:
        return jsonify({"id": knob_id, "value": knob_states[knob_id]}), 200
    return jsonify({"error": "Knob ID not found"}), 404

@app.route('/send-match', methods=['POST'])
def send_match():
    """
    Endpoint to receive and broadcast matched data via WebSocket.
    """
    try:
        # Parse JSON payload
        data = request.json

        # Validate payload structure
        if not all(key in data for key in ["matched_parameters", "matched_audio_data", "matched_spectrogram_data"]):
            return jsonify({"error": "Missing required keys in request payload"}), 400

        # Broadcast data via WebSocket
        socketio.emit('send_match', data)

        # Return success response
        return jsonify({"message": "Matched data broadcasted successfully"}), 200

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, debug=True)
