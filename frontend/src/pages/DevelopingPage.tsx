// src/pages/DevelopingPage.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { io } from 'socket.io-client';

const socket = io('http://127.0.0.1:5000');

type FileInfo = {
  filename: string;
  total: number;
  benign: number;
  udp_lag: number;
  syn_flood: number;
  udp_flood: number;
};

const DevelopingPage: React.FC = () => {
  const [file, setFile] = useState<File | null>(null);
  const [fileInfo, setFileInfo] = useState<FileInfo>({
    filename: 'Default',
    total: 0,
    benign: 0,
    udp_lag: 0,
    syn_flood: 0,
    udp_flood: 0,
  });

  // Choose file handler
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files?.[0];
    setFile(selected || null);
  };

  // Submit file handler
  const handleSubmit = async () => {
    if (!file) {
      alert("Please select a file first.");
      return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('http://127.0.0.1:5000/upload_3', formData);
      alert("Upload OK.");
    } catch (error: any) {
      alert('Upload failed: ' + (error.response?.data || error.message));
    }
  };

  // Listen for results
  useEffect(() => {
    socket.on('file_uploaded', (data: FileInfo) => {
      setFileInfo(data);
    });
  }, []);

  return (
    <div style={{ padding: '2rem' }}>
      <h1>Developing Page</h1>
      <input type="file" onChange={handleFileChange} />
      <button onClick={handleSubmit} style={{ marginLeft: '1rem' }}>Submit</button>

      <div style={{ marginTop: '2rem' }}>
        <p><strong>Filename:</strong> {fileInfo.filename}</p>
        <p><strong>Total Samples:</strong> {fileInfo.total}</p>
        <p><strong>Benign:</strong> {fileInfo.benign}</p>
        <p><strong>UDP Lag:</strong> {fileInfo.udp_lag}</p>
        <p><strong>SYN Flood:</strong> {fileInfo.syn_flood}</p>
        <p><strong>UDP Flood:</strong> {fileInfo.udp_flood}</p>
      </div>
    </div>
  );
};

export default DevelopingPage;
