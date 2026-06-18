import axios from 'axios';

const API_BASE = 'http://localhost:8000';

export const submitQuery = async (question) => {
  const res = await axios.post(`${API_BASE}/query`, { question });
  return res.data;
};

export const getQueries = async () => {
  const res = await axios.get(`${API_BASE}/queries`);
  return res.data;
};

export const getQueryDetail = async (id) => {
  const res = await axios.get(`${API_BASE}/queries/${id}`);
  return res.data;
};

export const getRetrievals = async (queryId) => {
  const res = await axios.get(`${API_BASE}/queries/${queryId}/retrievals`);
  return res.data;
};
