# ITP-Network-Forensics-Project

### Setup Environment

1. Create virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install required Python packages
```bash
pip install flask
pip install flask-socketio
pip install pandas
pip install matplotlib
pip install tensorflow
pip install scikit-learn
```

OR

```bash
pip install -r requirements.txt
```

3. Start webserver (Website will be accesible at http://127.0.0.1:5000/)
```bash
python3 main.py
```