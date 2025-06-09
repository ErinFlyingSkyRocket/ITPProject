import React from 'react';
import './styles.css';

const HomePage: React.FC = () => {
  return (
    <div className="page-container">
      <div className="page-header">Demo</div>

      <div className="card-container">
        <a href="/find">
          <button className="button-style">Find Optimum Architecture</button>
        </a>
        <a href="/evaluate">
          <button className="button-style">Evaluate Dataset</button>
        </a>

        <div id="box-container">
          <div className="box-style">
            Distributed Denial of Service (DDoS) Attack Dataset
            <br />
            Total: 877162
            <br /><br />
            Benign: 44811
            <br />
            UDP Lag: 77491
            <br />
            SYN Flood: 354860
            <br />
            UDP Flood: 400000
          </div>
        </div>

        <p style={{ textAlign: 'left', marginTop: '50px', color: 'blue' }}>
          Conv2D(32, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),
          MaxPooling2D((2, 2), strides=2), Dropout(0.50),
          Conv2D(64, (3, 3), activation='relu', padding='same'),
          MaxPooling2D((2, 2), strides=2), Dropout(0.25),
          Flatten(), Dense(1024, activation='relu'),
          Dropout(0.40), Dense(4),
          Adam optimizer with learning rate of 0.001
        </p>
        <p style={{ textAlign: 'left', color: 'blue', fontWeight: 'bold' }}>Acc: 0.8594</p>

        <p style={{ textAlign: 'left', marginTop: '50px', color: 'purple' }}>
          Conv2D(64, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),
          MaxPooling2D((2, 2), strides=2), Dropout(0.50),
          Conv2D(64, (7, 7), activation='relu', padding='same'),
          MaxPooling2D((2, 2), strides=2), Dropout(0.25),
          Flatten(), Dense(1024, activation='relu'),
          Dropout(0.40), Dense(4),
          Adam optimizer with learning rate of 0.05
        </p>
        <p style={{ textAlign: 'left', color: 'purple', fontWeight: 'bold' }}>Acc: 0.9788</p>
      </div>
    </div>
  );
};

export default HomePage;
