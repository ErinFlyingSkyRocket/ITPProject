// frontend/src/App.tsx
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import FindPage from './pages/FindPage';
import EvaluatePage from './pages/EvaluatePage';
import DevelopingPage from './pages/DevelopingPage';

const App = () => (
  <BrowserRouter>
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/find" element={<FindPage />} />
      <Route path="/evaluate" element={<EvaluatePage />} />
      <Route path="/developing" element={<DevelopingPage />} />
    </Routes>
  </BrowserRouter>
);

export default App;
