import { useParams, useLocation, useNavigate } from 'react-router-dom';
import ReactMarkdown from 'react-markdown';
import '../styles/ResultsPage.css';

export default function ResultsPage() {
  const { id } = useParams();
  const { state } = useLocation();
  const navigate = useNavigate();
  const result = state?.result;

  if (!result) {
    return (
      <div className="results-page">
        <div className="error-message">Result not found. <a href="/" onClick={() => navigate('/')}>Go back</a></div>
      </div>
    );
  }

  return (
    <div className="results-page">
      <div className="results-container">
        <button onClick={() => navigate('/')} className="back-btn">← Back to Query</button>

        {/* Original Question */}
        <div className="section">
          <h2>📝 Your Question</h2>
          <div className="question-box">
            <p>{result.question}</p>
          </div>
        </div>

        {/* Parsed Query Info */}
        {result.parsed_question && (
          <div className="section">
            <h2>📋 Parsed Query</h2>
            <div className="info-grid">
              {result.goal && <div className="info-item"><strong>Goal:</strong> {result.goal}</div>}
              {result.experience_level && <div className="info-item"><strong>Experience:</strong> {result.experience_level}</div>}
              {result.has_injury && <div className="info-item warning"><strong>⚠️ Injury Detected:</strong> Yes</div>}
            </div>
          </div>
        )}

        {/* Safety Flags */}
        {result.safety_flags && result.safety_flags.length > 0 && (
          <div className="section safety-section">
            <h2>⚠️ Safety Flags</h2>
            {result.safety_flags.map((flag, idx) => (
              <div key={idx} className={`safety-flag severity-${flag.severity}`}>
                <div className="flag-header">
                  <strong>{flag.concern}</strong>
                  <span className="severity-badge">{flag.severity.toUpperCase()}</span>
                </div>
                <p>{flag.recommendation}</p>
              </div>
            ))}
          </div>
        )}

        {/* Main Recommendation */}
        <div className="section recommendation-section">
          <h2>🤖 Recommendation</h2>
          <div className="recommendation-content">
            <ReactMarkdown>{result.recommendation || 'No recommendation generated.'}</ReactMarkdown>
          </div>
        </div>

        {/* Sources */}
        {result.sources_cited && result.sources_cited.length > 0 && (
          <div className="section sources-section">
            <h2>📚 Sources Cited</h2>
            <ul className="sources-list">
              {result.sources_cited.map((src, idx) => (
                <li key={idx}>{src}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Workflow Path */}
        {result.workflow_path && result.workflow_path.length > 0 && (
          <div className="section">
            <h2>🔄 Processing Pipeline</h2>
            <div className="workflow-path">
              {result.workflow_path.map((step, idx) => (
                <div key={idx}>
                  <span className="step">{step}</span>
                  {idx < result.workflow_path.length - 1 && <span className="arrow">→</span>}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error Info */}
        {result.error && (
          <div className="section">
            <div className="error-box">
              <strong>Error:</strong> {result.error}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
