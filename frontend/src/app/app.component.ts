import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  standalone: true,
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  logs = '';

  startTraining() {
    const eventSource = new EventSource('http://localhost:5000/start-training');
    this.logs = ''; // Clear old logs
    eventSource.onmessage = (event) => {
      this.logs += event.data + '\n';
    };
    eventSource.onerror = () => {
      eventSource.close();
    };
  }
}
