from flask import Flask, Response, jsonify
import subprocess
import sys

app = Flask(__name__)

@app.route('/start-training')
def start_training():
    def generate():
        try:
            process = subprocess.Popen(
                [sys.executable, 'rl-cnn/rl-cnn.py'],  # Uses correct venv python
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            for line in iter(process.stdout.readline, ''):
                print("DEBUG LINE:", line.strip())  # Terminal debug
                yield f"data: {line.strip()}\n\n"    # SSE format
            process.stdout.close()
            process.wait()
        except Exception as e:
            yield f"data: ERROR: {str(e)}\n\n"
            print("ERROR:", e)

    return Response(generate(), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True, port=5000)
