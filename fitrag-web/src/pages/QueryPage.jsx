import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { submitQuery } from '../api/client';
import '../styles/QueryPage.css';

export default function QueryPage() {
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) {
      setError('Please enter a question');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await submitQuery(question);
      navigate(`/results/${result.id}`, { state: { result } });
    } catch (err) {
      setError(err.response?.data?.detail || 'Error submitting query. Make sure the backend is running.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-page">
      <div className="query-container">
        <div className="header">
          <h1>🏋️ FitRAG</h1>
          <p>Evidence-Based Fitness & Nutrition Intelligence</p>
        </div>

        <form onSubmit={handleSubmit} className="query-form">
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Ask your fitness or nutrition question...&#10;&#10;Example: I'm 28, vegan, have a bad knee, and want to build muscle training 4 days a week"
            rows="6"
            disabled={loading}
            className="query-input"
          />

          {error && <div className="error-message">{error}</div>}

          <button type="submit" disabled={loading} className="submit-btn">
            {loading ? '⏳ Processing...' : '🚀 Get Recommendation'}
          </button>
        </form>

        <div className="info">
          <p>Our system will parse your question, check for safety concerns, and retrieve evidence-based recommendations from research papers.</p>
        </div>
      </div>
    </div>
  );
}
