import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import './styles.css';

const socket = io('http://127.0.0.1:5000');

const EvaluatePage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState('');
  const [evaluationResult, setEvaluationResult] = useState('');
  const [showMatrix, setShowMatrix] = useState(false);
  const [progress, setProgress] = useState(0);
  const [fileStats, setFileStats] = useState<null | {
    filename: string;
    total: number;
    benign: number;
    udp_lag: number;
    syn_flood: number;
    udp_flood: number;
  }>(null);

  useEffect(() => {
    socket.on('file_uploaded', (data) => {
      setFileStats(data);
      setUploadStatus('File uploaded. Ready to evaluate.');
      setEvaluationResult('');
      setShowMatrix(false);
    });

    socket.on('evaluation_done', (data) => {
      setEvaluationResult(`Overall Acc: ${data.accuracy}`);
      setProgress(0);
    });

    socket.on('show_confusion_matrix', () => {
      setShowMatrix(true);
    });

    return () => {
      socket.off('file_uploaded');
      socket.off('evaluation_done');
      socket.off('show_confusion_matrix');
    };
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploaded = e.target.files?.[0] || null;
    setFile(uploaded);
    setUploadStatus('');
    setFileStats(null);
    setEvaluationResult('');
    setShowMatrix(false);
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    await fetch('http://127.0.0.1:5000/upload_2', {
      method: 'POST',
      body: formData,
    }).then(response => {
      if (!response.ok && response.status !== 204) {
        throw new Error('Upload failed');
      }
    });
  };

  const handleEvaluate = () => {
    socket.emit('Evaluate_2');
    startProgressBar();
    setEvaluationResult('');
    setShowMatrix(false);
  };

  const startProgressBar = () => {
    let percent = 0;
    const interval = setInterval(() => {
      percent += 2;
      setProgress(percent);
      if (percent >= 100) clearInterval(interval);
    }, 150);
  };

  return (
    <div className="page-container">
      <div className="page-header">Evaluate Dataset</div>

      <div className="card-container">
        <a href="/find"><button className="button-style">Find Optimum Architecture</button></a>
        <a href="/"><button className="button-style">Home</button></a>

        <h1>Upload Testing Dataset</h1>
        <p style={{ textAlign: 'left' }}>
          <b>Trained Optimum Architecture:</b><br />
          Conv2D(64, (5, 5), activation='relu', input_shape=(9, 9, 1), padding='same'),<br />
          MaxPooling2D((2, 2), strides=2), Dropout(0.50), Conv2D(64, (7, 7), activation='relu', padding='same'),<br />
          MaxPooling2D((2, 2), strides=2), Dropout(0.25), Flatten(), Dense(1024, activation='relu'),<br />
          Dropout(0.40), Dense(4), Adam optimizer with learning rate of 0.05
        </p>

        <input type="file" onChange={handleFileChange} />
        <button className="button-style upload-button" onClick={handleUpload} disabled={!file}>
          Upload
        </button>

        {uploadStatus && <p className="upload-status">{uploadStatus}</p>}

        {fileStats && (
          <div className="box-style">
            📄 File: {fileStats.filename}
            <br />Total: {fileStats.total}
            <br /><br />Benign: {fileStats.benign}
            <br />UDP Lag: {fileStats.udp_lag}
            <br />SYN Flood: {fileStats.syn_flood}
            <br />UDP Flood: {fileStats.udp_flood}
            <br /><br />
            <button className="button-style" onClick={handleEvaluate}>Evaluate</button>
          </div>
        )}

        {progress > 0 && (
          <div className="progress-container">
            <div className="progress-bar" style={{ width: `${progress}%` }} />
          </div>
        )}

        {evaluationResult && <div className="result-style">{evaluationResult}</div>}

        {showMatrix && (
          <div className="confusion-container">
            <img
              src={`/get_confusion_matrix?ts=${new Date().getTime()}`}
              style={{ maxWidth: '100%', height: 'auto', width: '500px' }}
              alt="Confusion Matrix"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default EvaluatePage;
