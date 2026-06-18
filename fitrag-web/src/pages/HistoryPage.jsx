import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getQueries, getQueryDetail } from '../api/client';
import '../styles/HistoryPage.css';

export default function HistoryPage() {
  const [queries, setQueries] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchQueries();
  }, []);

  const fetchQueries = async () => {
    try {
      setLoading(true);
      const data = await getQueries();
      setQueries(data);
    } catch (err) {
      setError('Failed to load query history. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleQueryClick = async (queryId) => {
    try {
      const detail = await getQueryDetail(queryId);
      navigate(`/results/${queryId}`, { state: { result: detail } });
    } catch (err) {
      setError('Failed to load query details');
      console.error(err);
    }
  };

  return (
    <div className="history-page">
      <div className="history-container">
        <h1>📜 Query History</h1>

        {error && <div className="error-message">{error}</div>}

        {loading ? (
          <div className="loading">Loading...</div>
        ) : queries.length === 0 ? (
          <div className="empty-state">
            <p>No queries yet. <a href="/" onClick={() => navigate('/')}>Ask a question</a> to get started!</p>
          </div>
        ) : (
          <div className="queries-list">
            {queries.map((q) => (
              <div
                key={q.id}
                className="query-card"
                onClick={() => handleQueryClick(q.id)}
              >
                <div className="query-card-header">
                  <h3>Query #{q.id}</h3>
                  <span className="date">{new Date(q.created_at).toLocaleString()}</span>
                </div>
                <p className="query-text">{q.question.substring(0, 150)}...</p>
                <div className="query-meta">
                  {q.goal && <span className="badge">{q.goal}</span>}
                  {q.has_injury && <span className="badge warning">⚠️ Injury</span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
