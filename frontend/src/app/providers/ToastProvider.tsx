import { createContext, useContext, ReactNode } from 'react';
import { useToast, ToastType } from '@/hooks/useToast';
import ToastContainer from '@/components/Toast/ToastContainer';

interface ToastApi {
  showToast: (message: string, type?: ToastType, duration?: number) => void;
}

const ToastContext = createContext<ToastApi | undefined>(undefined);

interface ToastProviderProps {
  children: ReactNode;
}

function ToastProvider({ children }: ToastProviderProps) {
  const { toasts, showToast, hideToast } = useToast();

  return (
    <ToastContext.Provider value={{ showToast }}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={hideToast} />
    </ToastContext.Provider>
  );
}

export function useToastApi(): ToastApi {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToastApi must be used within ToastProvider');
  return ctx;
}

export default ToastProvider;
