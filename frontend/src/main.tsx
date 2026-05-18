import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import ErrorBoundary from '@/components/ErrorBoundary';
import AudioProvider from '@/app/providers/AudioProvider';
import ToastProvider from '@/app/providers/ToastProvider';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <ToastProvider>
        <AudioProvider>
          <App />
        </AudioProvider>
      </ToastProvider>
    </ErrorBoundary>
  </React.StrictMode>,
);
