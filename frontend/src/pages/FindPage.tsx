// frontend/src/pages/FindPage.tsx
import React from 'react';

const FindPage = () => {
  return (
    <div className="container">
      <h1>Architecture Discovery</h1>
      <p>Click on the button to extract the model architecture:</p>
      <button
        onClick={() => alert("Pretend this fetches architecture")}
        className="btn btn-primary"
      >
        Get Architecture
      </button>

      <div style={{ marginTop: "2rem" }}>
        <h3>Extracted Architecture:</h3>
        <pre>
{`Layer (type)                 Output Shape              Param #
=================================================================
conv2d_1 (Conv2D)            (None, 26, 26, 32)        320
max_pooling2d_1 (MaxPooling2D) (None, 13, 13, 32)       0
flatten (Flatten)            (None, 5408)              0
dense_1 (Dense)              (None, 128)               692352
dense_2 (Dense)              (None, 10)                1290
=================================================================
Total params: 693,962
Trainable params: 693,962
Non-trainable params: 0`}
        </pre>
      </div>
    </div>
  );
};

export default FindPage;
