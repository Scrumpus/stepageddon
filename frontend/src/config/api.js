import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
export const generateStepsFromFile = async (file, difficulty) => {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('difficulty', difficulty);
  
  const response = await api.post('/api/generate-steps', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  
  return response.data;
};

export const generateStepsFromUrl = async (url, difficulty) => {
  const response = await api.post('/api/generate-steps-url', {
    url,
    difficulty,
  });
  
  return response.data;
};

export const getAudioUrl = (audioPath) => {
  return `${API_BASE_URL}${audioPath}`;
};
