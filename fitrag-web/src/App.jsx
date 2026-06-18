import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import QueryPage from './pages/QueryPage';
import ResultsPage from './pages/ResultsPage';
import HistoryPage from './pages/HistoryPage';
import './App.css';

function App() {
  return (
    <BrowserRouter>
      <nav className="navbar">
        <div className="nav-content">
          <Link to="/" className="nav-brand">
            🏋️ FitRAG
          </Link>
          <div className="nav-links">
            <Link to="/" className="nav-link">Home</Link>
            <Link to="/history" className="nav-link">History</Link>
          </div>
        </div>
      </nav>

      <main className="main-content">
        <Routes>
          <Route path="/" element={<QueryPage />} />
          <Route path="/results/:id" element={<ResultsPage />} />
          <Route path="/history" element={<HistoryPage />} />
        </Routes>
      </main>

      <footer className="footer">
        <p>FitRAG © 2024 - Evidence-Based Fitness Intelligence</p>
      </footer>
    </BrowserRouter>
  );
}

export default App;
