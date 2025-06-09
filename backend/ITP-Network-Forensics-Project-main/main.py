from flask import Flask, render_template, request, send_file, redirect, url_for
from flask_socketio import SocketIO, emit
import pandas as pd
import time
import matplotlib.pyplot as plt
import numpy as np
import io
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")
app.debug = True
list_classes = ["Benign", "UDP Lag", "SYN Flood", "UDP Flood"]

temp_data = {}
temp_df = pd.DataFrame()
global_data = None

@app.route('/developing')
def developing():
    default_file_info = {
        'filename': 'Default',
        'total': 0,
        'benign': 0,
        'udp_lag': 0,
        'syn_flood': 0,
        'udp_flood': 0
    }
    return render_template('developing.html', default_file=default_file_info)

@app.route('/upload_3', methods=['POST'])
def upload_3():
    global temp_df

    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        df = pd.read_csv(file, header=None)
        temp_df = df  # Store the DataFrame in a global variable for later use

        filename = file.filename
        data = df.to_numpy()
        data_test = data[:,:-1]
        test_labels = data[:,-1]

        classes, samples_each_class = np.unique(test_labels, return_counts=True)
        class_counts = dict(zip(classes.astype(int), samples_each_class))

        temp_data['filename'] = filename
        temp_data['total'] = len(test_labels)
        temp_data['benign'] = class_counts.get(0, 0)
        temp_data['udp_lag'] = class_counts.get(1, 0)
        temp_data['syn_flood'] = class_counts.get(2, 0)
        temp_data['udp_flood'] = class_counts.get(3, 0)

        socketio.emit('file_uploaded', {
            'filename': filename,
            'total': temp_data['total'],
            'benign': int(temp_data['benign']),
            'udp_lag': int(temp_data['udp_lag']),
            'syn_flood': int(temp_data['syn_flood']),
            'udp_flood': int(temp_data['udp_flood'])
        })

        return '', 204
    except Exception as e:
        return f"Error reading file: {e}", 500


@socketio.on('convert')
def handle_convert():
    global temp_df  # Use the global DataFrame stored in upload_3

    socketio.emit('clear_previous_results')

    if temp_df.empty:
        emit('error', {'message': 'No file uploaded or file is empty.'})
        return
    
    try:
        # Separate features and labels
        X = temp_df.iloc[:, :-1]
        y = temp_df.iloc[:, -1]

        # Normalize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, stratify=y, random_state=42
        )

        # Combine and save
        train_df = pd.DataFrame(X_train)
        train_df['label'] = y_train.values
        test_df = pd.DataFrame(X_test)
        test_df['label'] = y_test.values

        # Save as CSV strings
        train_csv = train_df.to_csv(index=False, header=False)
        test_csv = test_df.to_csv(index=False, header=False)

        # Count classes
        def count_classes(labels):
            classes, counts = np.unique(labels, return_counts=True)
            return {int(cls): int(count) for cls, count in zip(classes, counts)}

        emit('conversion_result', {
            'train_counts': count_classes(y_train),
            'test_counts': count_classes(y_test),
            'input_counts': count_classes(y),
        })

        # Emit files
        socketio.emit('download_csv', {
            'filename': 'train_data.csv',
            'data': train_csv
        })
        socketio.emit('download_csv', {
            'filename': 'test_data.csv',
            'data': test_csv
        })

    except Exception as e:
        emit('error', {'message': f'Error during conversion: {e}'})


@app.route('/find')
def find():
    default_file_info = {
        'filename': 'Default',
        'total': 877162,
        'benign': 44811,
        'udp_lag': 77491,
        'syn_flood': 354860,
        'udp_flood': 400000
    }
    return render_template('find.html', default_file=default_file_info)


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        df = pd.read_csv(file)
        filename = file.filename
        data = df.to_numpy()
        data_test = data[:,:-1]
        test_labels = data[:,-1]

        classes, samples_each_class = np.unique(test_labels, return_counts=True)
        class_counts = dict(zip(classes.astype(int), samples_each_class))

        temp_data['filename'] = filename
        temp_data['total'] = len(test_labels)
        temp_data['benign'] = class_counts.get(0, 0)
        temp_data['udp_lag'] = class_counts.get(1, 0)
        temp_data['syn_flood'] = class_counts.get(2, 0)
        temp_data['udp_flood'] = class_counts.get(3, 0)

        socketio.emit('file_uploaded', {
            'filename': filename,
            'total': temp_data['total'],
            'benign': int(temp_data['benign']),
            'udp_lag': int(temp_data['udp_lag']),
            'syn_flood': int(temp_data['syn_flood']),
            'udp_flood': int(temp_data['udp_flood'])
        })

        return '', 204
    except Exception as e:
        return f"Error reading file: {e}", 500


@socketio.on('find')
def handle_find():
    numbers = ["Conv2D(32, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (3, 3), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.001", "Conv2D(64, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (3, 3), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.001", "Conv2D(64, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (3, 3), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.05","Conv2D(64, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (7, 7), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.05","Conv2D(64, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (7, 7), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.05","Conv2D(64, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (7, 7), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.005","Conv2D(64, (3, 3), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (7, 7), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.0005","Conv2D(32, (7, 7), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (7, 7), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.0005","Conv2D(32, (7, 7), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (7, 7), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.001","Conv2D(32, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.50),Conv2D(64, (3, 3), activation='relu',  padding='same'),MaxPooling2D((2, 2), strides=2),Dropout(0.25),Flatten(),Dense(1024, activation='relu'),Dropout(0.40),Dense(4),Adam optimizer with learning rate of 0.001"]
    letters = [0.8594, 0.8594, 0.9783, 0.9788,0.9787,0.9761,0.8594,0.8594,0.8594,0.8594]

    socketio.emit('clear_previous_results', data, broadcast=True)

    for i in range(len(numbers)):
        emit('find_step', {
            'iteration': i,
            'number': numbers[i],
            'letter': letters[i],
            'wait_time': 20
        })
        time.sleep(20)


@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html', greeting="Hello from Flask!")

@app.route('/upload_2', methods=['POST'])
def upload_2():
    global global_data
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    try:
        df = pd.read_csv(file)
        filename = file.filename

        data = df.to_numpy()
        global_data = data
        data_test = data[:,:-1]
        test_labels = data[:,-1]

        classes, samples_each_class = np.unique(test_labels, return_counts=True)
        class_counts = dict(zip(classes.astype(int), samples_each_class))

        temp_data['filename'] = filename
        temp_data['total'] = len(test_labels)
        temp_data['benign'] = class_counts.get(0, 0)
        temp_data['udp_lag'] = class_counts.get(1, 0)
        temp_data['syn_flood'] = class_counts.get(2, 0)
        temp_data['udp_flood'] = class_counts.get(3, 0)

        socketio.emit('file_uploaded', {
            'filename': filename,
            'total': temp_data['total'],
            'benign': int(temp_data['benign']),
            'udp_lag': int(temp_data['udp_lag']),
            'syn_flood': int(temp_data['syn_flood']),
            'udp_flood': int(temp_data['udp_flood'])
        })

        return '', 204
    except Exception as e:
        return f"Error reading file: {e}", 500


@app.route('/get_confusion_matrix')
def get_confusion_matrix():
    global conf_matrix_global

    if conf_matrix_global is None:
        return "No matrix available", 400

    fig, ax = plt.subplots(figsize=(6, 5))

    cax = ax.matshow(conf_matrix_global, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticklabels(["", "Benign", "UDP Lag", "SYN Flood", "UDP Flood"])
    ax.set_yticklabels(["", "Benign", "UDP Lag", "SYN Flood", "UDP Flood"])

    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(conf_matrix_global[i, j]), ha='center', va='center', color='orange', fontsize=20)

    img_io = io.BytesIO()
    plt.savefig(img_io, format='png', bbox_inches='tight')
    img_io.seek(0)
    return send_file(img_io, mimetype='image/png')

def padding_reshape(data):
    cnn_input_shape = 81
    num_padding = 81 - 59
    left_num_padding = num_padding // 2
    right_num_padding = num_padding - left_num_padding
    padded_data = np.pad(data, ((0, 0), (left_num_padding, right_num_padding)), mode='constant', constant_values=0)
    padded_data = padded_data.reshape((-1, 9, 9))
    padded_data = np.expand_dims(padded_data, axis=-1)
    return padded_data

@socketio.on('Evaluate_2')
def evaluate_file():
    global global_data, conf_matrix_global

    data_test = global_data[:,:-1]
    test_labels = global_data[:,-1]
    test_images = padding_reshape(data_test)
    model = tf.keras.models.load_model('my_model.h5')
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0)
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    conf_matrix = tf.math.confusion_matrix(test_labels, predicted_labels)
    conf_matrix = conf_matrix.numpy()

    conf_matrix_global = conf_matrix
    accuracy = round(test_acc, 4)

    time.sleep(10)

    socketio.emit('evaluation_done', {
        'accuracy': accuracy
    })

    time.sleep(3)
    socketio.emit('show_confusion_matrix')

if __name__ == '__main__':
    socketio.run(app, allow_unsafe_werkzeug=True)
