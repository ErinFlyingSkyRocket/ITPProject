from flask import Flask, Response, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route('/start-training')
def start_training():
    try:
        print("🔥 Flask endpoint hit, attempting to run rl-cnn.py", flush=True)

        result = subprocess.run(
            [sys.executable, 'rl-cnn/rl-cnn.py'],
            capture_output=True,
            text=True,
            check=True
        )

        return jsonify({
            "status": "success",
            "output": result.stdout[-500:]  # just last 500 chars
        })

    except subprocess.CalledProcessError as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "stderr": e.stderr
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
