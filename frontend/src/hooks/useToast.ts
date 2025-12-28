/**
 * Toast notification hook
 * Replaces alert() with better UX
 */

import { useState, useCallback } from 'react';

export type ToastType = 'success' | 'error' | 'info' | 'warning';

export interface Toast {
  id: string;
  type: ToastType;
  message: string;
  duration: number;
}

let toastIdCounter = 0;

export function useToast() {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const showToast = useCallback((
    message: string,
    type: ToastType = 'info',
    duration: number = 5000
  ) => {
    const id = `toast-${toastIdCounter++}`;
    const newToast: Toast = { id, type, message, duration };

    setToasts((prev) => [...prev, newToast]);

    // Auto-dismiss after duration
    if (duration > 0) {
      setTimeout(() => {
        hideToast(id);
      }, duration);
    }
  }, []);

  const hideToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  return {
    toasts,
    showToast,
    hideToast,
  };
}
