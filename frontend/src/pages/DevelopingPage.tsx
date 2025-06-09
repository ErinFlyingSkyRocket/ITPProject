import React, { useEffect } from 'react';
import './styles.css';
import { io } from 'socket.io-client';

const DevelopingPage: React.FC = () => {
  useEffect(() => {
    const socket = io('http://localhost:5000');

    const boxContainer = document.getElementById('box-container');
    const secondBoxContainer = document.getElementById('secondbox-container');
    const findBtnContainer = document.getElementById('find-btn-container');
    const uploadForm = document.getElementById('upload-form') as HTMLFormElement;

    function renderFileBox(data: any) {
      if (!boxContainer) return;
      boxContainer.innerHTML = '';
      const fileBox = document.createElement('div');
      fileBox.className = 'box';
      fileBox.textContent = `📄 File: ${data.filename}\nTotal:\t${data.total}\n\nBenign:\t${data.benign}\nUDP Lag:\t${data.udp_lag}\nSYN Flood:\t${data.syn_flood}\nUDP Flood:\t${data.udp_flood}`;
      boxContainer.appendChild(fileBox);
    }

    function renderConvertButton() {
      if (!findBtnContainer) return;
      findBtnContainer.innerHTML = '';
      const btn = document.createElement('button');
      btn.textContent = 'Convert';
      btn.id = 'convert';
      btn.onclick = () => socket.emit('convert');
      findBtnContainer.appendChild(btn);
    }

    function createProgressBar(waitTime: number, callback: () => void) {
      if (!secondBoxContainer) return;
      const container = document.createElement('div');
      container.className = 'progress-container';
      const bar = document.createElement('div');
      bar.className = 'progress-bar';
      container.appendChild(bar);
      secondBoxContainer.appendChild(container);

      let percent = 0;
      const interval = setInterval(() => {
        percent += 1;
        bar.style.width = percent + '%';
        if (percent >= 100) {
          clearInterval(interval);
          container.remove();
          callback();
        }
      }, waitTime * 10);
    }

    socket.on('file_uploaded', (data: any) => {
      renderFileBox(data);
      renderConvertButton();
    });

    socket.on('clear_previous_results', () => {
      if (secondBoxContainer) secondBoxContainer.innerHTML = '';
    });

    socket.on('error', (error: any) => {
      if (!secondBoxContainer) return;
      const errorBox = document.createElement('div');
      errorBox.className = 'secondbox';
      errorBox.textContent = `Error: ${error['message']}`;
      errorBox.style.backgroundColor = '#FFCCCB';
      secondBoxContainer.appendChild(errorBox);
    });

    socket.on('conversion_result', (data: any) => {
      if (!secondBoxContainer) return;
      secondBoxContainer.innerHTML = '';

      const resultBox = document.createElement('div');
      resultBox.className = 'secondbox';

      const classNames: Record<string, string> = {
        '0': 'Benign',
        '1': 'UDP Lag',
        '2': 'SYN Flood',
        '3': 'UDP Flood',
      };

      const table = document.createElement('table');
      table.style.width = '100%';
      table.style.borderCollapse = 'collapse';

      const headerRow = document.createElement('tr');
      ['Class', 'Input Count', 'Train Count', 'Test Count'].forEach((header) => {
        const th = document.createElement('th');
        th.textContent = header;
        th.style.border = '1px solid #333';
        th.style.padding = '8px';
        th.style.backgroundColor = '#d5f5e3';
        headerRow.appendChild(th);
      });
      table.appendChild(headerRow);

      const allLabels = new Set([
        ...Object.keys(data.input_counts),
        ...Object.keys(data.train_counts),
        ...Object.keys(data.test_counts),
      ]);

        [...Array.from(allLabels)].sort().forEach((label) => {
        const row = document.createElement('tr');

        const tdClass = document.createElement('td');
        tdClass.textContent = classNames[label] || `Class ${label}`;
        const tdInput = document.createElement('td');
        tdInput.textContent = data.input_counts[label] ?? '0';
        const tdTrain = document.createElement('td');
        tdTrain.textContent = data.train_counts[label] ?? '0';
        const tdTest = document.createElement('td');
        tdTest.textContent = data.test_counts[label] ?? '0';

        [tdClass, tdInput, tdTrain, tdTest].forEach((td) => {
          td.style.border = '1px solid #333';
          td.style.padding = '6px';
          td.style.textAlign = 'center';
          row.appendChild(td);
        });

        table.appendChild(row);
      });

      const header = document.createElement('h3');
      header.textContent = 'Converted Data';
      resultBox.appendChild(header);
      resultBox.appendChild(table);
      secondBoxContainer.appendChild(resultBox);
    });

    socket.on('download_csv', (payload: any) => {
      const filename = payload.filename || 'download.csv';
      const csvContent = payload.data;
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
      const link = document.createElement('a');
      if (link.download !== undefined) {
        const url = URL.createObjectURL(blob);
        link.setAttribute('href', url);
        link.setAttribute('download', filename);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      } else {
        alert('CSV download is not supported in this browser.');
      }
    });

    const fileInput = document.getElementById('file-input') as HTMLInputElement;
    uploadForm?.addEventListener('submit', function (e) {
      e.preventDefault();
      if (secondBoxContainer) secondBoxContainer.innerHTML = '';
      const formData = new FormData();
      const file = fileInput?.files?.[0];
      if (file) {
        formData.append('file', file);
        fetch('/upload_3', {
          method: 'POST',
          body: formData,
        });
      }
    });

    // Initial render from defaultFileInfo
    const defaultFileInfo = (window as any).defaultFileInfo;
    if (defaultFileInfo) {
      renderFileBox(defaultFileInfo);
      renderConvertButton();
    }
    return () => {
        socket.disconnect();
    };
    }, []);


  return (
    <div className="page-container">
        <div className="page-header">Train Test Split</div>
        <div className="card-container">
        <div style={{ marginBottom: '20px' }}>
            <a href="/evaluate">
            <button className="button-style">Evaluate Dataset</button>
            </a>
            <a href="/">
            <button className="button-style upload-button">Home</button>
            </a>
        </div>

        <h1>Upload Dataset</h1>
        <form id="upload-form" encType="multipart/form-data">
            <input type="file" id="file-input" name="file" required />
            <button id="upload_3" type="submit" className="button-style upload-button">Upload</button>
        </form>

        <div id="box-container" className="box-style"></div>
        <div id="find-btn-container"></div>
        <div id="secondbox-container" className="confusion-container"></div>
        </div>
    </div>
    );
};

export default DevelopingPage;
