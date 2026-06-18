# FitRAG Web - React Frontend

This is the React web interface for FitRAG, a personalized fitness & nutrition intelligence system powered by RAG.

## Features

- 🏋️ Query interface for asking fitness questions
- 📜 Query history view
- 📊 Detailed results with safety flags and recommendations
- 📚 Source citation display
- 🔄 Workflow visualization

## Installation

```bash
npm install
```

## Running Locally

Make sure the FastAPI backend is running first:

```bash
# Terminal 1: Start FastAPI backend
cd ../
python -m app.api.main
# Server will run at http://localhost:8000
```

Then in another terminal, start the React app:

```bash
# Terminal 2: Start React frontend
npm start
```

The app will open at `http://localhost:3000` (or `http://localhost:5173` if using Vite dev server).

## Building for Production

```bash
npm run build
```

This creates an optimized production build in the `dist/` directory.

## Architecture

- **Frontend:** React 18 + React Router + Axios
- **Styling:** CSS with responsive design
- **API Communication:** REST API calls to FastAPI backend
- **Build Tool:** Vite (modern and fast)

## Project Structure

```
src/
├── pages/
│   ├── QueryPage.jsx      # Main query input page
│   ├── ResultsPage.jsx    # Display results and recommendations
│   └── HistoryPage.jsx    # View past queries
├── api/
│   └── client.js          # API client functions
├── styles/
│   ├── QueryPage.css
│   ├── ResultsPage.css
│   └── HistoryPage.css
├── App.jsx                # Main app component with routing
├── main.jsx               # React entry point
└── index.css              # Global styles
```

## Environment

The app connects to the FastAPI backend at `http://localhost:8000` by default. To change this, edit `src/api/client.js`.

## Dependencies

- `react` - UI library
- `react-dom` - React DOM rendering
- `react-router-dom` - Client-side routing
- `axios` - HTTP client
- `react-markdown` - Markdown rendering for recommendations
- `vite` - Build tool

## License

Same as FitRAG project
