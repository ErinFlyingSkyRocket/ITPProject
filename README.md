# ML Dashboard (Flask + Angular)

This project consists of a Flask backend with a machine learning model and an Angular frontend dashboard.

---

## Backend Setup (Flask)

1. Navigate to the `backend/` folder:

   ```
   cd backend
   ```

2. Create and activate a virtual environment:

   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   ```

3. Install dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Start the Flask server:

   ```
   python main.py
   ```

---

## Frontend Setup (Angular)

1. Navigate to the `frontend/` folder:

   ```
   cd frontend
   ```

2. Install dependencies:

   ```
   npm install
   ```

3. Start the Angular dev server:

   ```
   ng serve
   ```

---

## Notes

- Flask runs on: `http://localhost:5000`
- Angular runs on: `http://localhost:4200`
- `.venv/` and `node_modules/` are excluded via `.gitignore`
