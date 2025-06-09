// frontend/src/pages/HomePage.tsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import './HomePage.css';

const HomePage: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div>
      <div className="header">Demo</div>

      <div className="container">
        <button onClick={() => navigate('/find')}>Find Optimum Architecture</button>
        <button onClick={() => navigate('/evaluate')}>Evaluate Dataset</button>

        <div className="box">
          Distributed Denial of Service (DDoS) Attack Dataset<br />
          Total: 877162<br /><br />
          Benign: 44811<br />
          UDP Lag: 77491<br />
          SYN Flood: 354860<br />
          UDP Flood: 400000
        </div>

        <p style={{ textAlign: 'left', marginTop: 50 }}>
          <span style={{ color: 'blue' }}>
            Conv2D(32, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),
            MaxPooling2D((2, 2), strides=2), Dropout(0.50),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), strides=2), Dropout(0.25),
            Flatten(), Dense(1024, activation='relu'), Dropout(0.40),
            Dense(4), Adam optimizer with learning rate of 0.001
          </span>
        </p>
        <p style={{ textAlign: 'left' }}>
          <span style={{ color: 'blue', fontWeight: 'bold' }}>Acc: 0.8594</span>
        </p>

        <p style={{ textAlign: 'left', marginTop: 50 }}>
          <span style={{ color: 'purple' }}>
            Conv2D(64, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),
            MaxPooling2D((2, 2), strides=2), Dropout(0.50),
            Conv2D(64, (7, 7), activation='relu', padding='same'),
            MaxPooling2D((2, 2), strides=2), Dropout(0.25),
            Flatten(), Dense(1024, activation='relu'), Dropout(0.40),
            Dense(4), Adam optimizer with learning rate of 0.05
          </span>
        </p>
        <p style={{ textAlign: 'left' }}>
          <span style={{ color: 'purple', fontWeight: 'bold' }}>Acc: 0.9788</span>
        </p>
      </div>
    </div>
  );
};

export default HomePage;
