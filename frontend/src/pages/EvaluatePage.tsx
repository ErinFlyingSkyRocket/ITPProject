// frontend/src/pages/EvaluatePage.tsx
import React, { useState } from 'react';

const EvaluatePage = () => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const handleEvaluate = () => {
    if (!selectedFile) {
      alert("Please select a file first.");
      return;
    }

    // Simulated prediction output
    setTimeout(() => {
      setResult("Prediction: The uploaded traffic is classified as DDoS");
    }, 1000);
  };

  return (
    <div className="container">
      <h1>Evaluate Model</h1>
      <p>Select a test file (e.g., CSV or image) to evaluate the model:</p>

      <input type="file" onChange={handleFileChange} />
      <br /><br />
      <button className="btn btn-success" onClick={handleEvaluate}>
        Evaluate
      </button>

      {result && (
        <div style={{ marginTop: "2rem" }}>
          <h3>Result:</h3>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
};

export default EvaluatePage;
