import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './styles.css';

const socket = io('http://localhost:5000');

const FindPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [fileStats, setFileStats] = useState<null | {
    filename: string;
    total: number;
    benign: number;
    udp_lag: number;
    syn_flood: number;
    udp_flood: number;
  }>({
    filename: 'Default',
    total: 877162,
    benign: 44811,
    udp_lag: 77491,
    syn_flood: 354860,
    udp_flood: 400000
  });

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploaded = e.target.files?.[0] || null;
    setFile(uploaded);
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    });
  };

  useEffect(() => {
    socket.on('file_uploaded', (data) => {
      setFileStats(data);
    });

    return () => {
      socket.off('file_uploaded');
    };
  }, []);

  const [findResults, setFindResults] = useState<Array<{
    architecture: string;
    progress: number;
    showProgress: boolean;
    accuracy?: string;
  }>>([]);

  const handleFind = () => {
    socket.emit('find');
  };

  useEffect(() => {
    socket.on('find_step', (data) => {
      const { iteration, number, letter, wait_time } = data;

      setFindResults(prev => [
        ...prev,
        {
          architecture: `Episode 0, Iteration ${iteration}\n${number}\n`,
          progress: 0,
          showProgress: true,
        }
      ]);

      let progress = 0;
      const interval = setInterval(() => {
        progress += 1;
        setFindResults(prev => {
          const updated = [...prev];
          updated[iteration] = {
            ...updated[iteration],
            progress,
          };
          return updated;
        });

        if (progress >= 100) {
          clearInterval(interval);
          setFindResults(prev => {
            const updated = [...prev];
            updated[iteration] = {
              ...updated[iteration],
              showProgress: false,
              accuracy: letter,
            };
            return updated;
          });
        }
      }, wait_time * 10);
    });

    socket.on('clear_previous_results', () => {
      setFindResults([]);
    });

    return () => {
      socket.off('find_step');
      socket.off('clear_previous_results');
    };
  }, []);

  return (
    <div className="page-container">
      <div className="page-header">Find Optimum Architecture</div>

      <div className="card-container">
        <a href="/evaluate"><button className="button-style">Evaluate Dataset</button></a>
        <a href="/"><button className="button-style">Home</button></a>

        <h1>Upload Dataset</h1>
        <input type="file" onChange={handleFileChange} />
        <button className="button-style upload-button" onClick={handleUpload} disabled={!file}>
          Upload
        </button>

        {fileStats && (
          <div className="box-style">
            📄 File: {fileStats.filename}
            <br />Total: {fileStats.total}
            <br /><br />Benign: {fileStats.benign}
            <br />UDP Lag: {fileStats.udp_lag}
            <br />SYN Flood: {fileStats.syn_flood}
            <br />UDP Flood: {fileStats.udp_flood}
          </div>
        )}

        <p style={{ textAlign: 'left' }}>
          Initial Architecture: <b>
            Conv2D(32, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),
            MaxPooling2D((2, 2), strides=2), Dropout(0.50),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), strides=2), Dropout(0.25),
            Flatten(), Dense(1024, activation='relu'),
            Dropout(0.40), Dense(4), Adam optimizer with learning rate of 0.001
          </b>
        </p>

        {fileStats && (
          <button className="button-style" onClick={handleFind}>Find</button>
        )}

        <div style={{ marginTop: '20px' }}>
          {findResults.map((result, idx) => (
            <div key={idx} className="result-style">
              {result.architecture}
              {result.showProgress ? (
                <div className="progress-container">
                  <div className="progress-bar" style={{ width: `${result.progress}%` }} />
                </div>
              ) : (
                <div><b>Acc: {result.accuracy}</b></div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default FindPage;
