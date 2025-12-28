/**
 * Individual Toast notification component
 */

import { useEffect, useState } from 'react';
import { X, CheckCircle, AlertCircle, Info, AlertTriangle } from 'lucide-react';
import type { Toast as ToastType } from '@/hooks/useToast';

interface ToastProps {
  toast: ToastType;
  onDismiss: (id: string) => void;
}

const toastStyles = {
  success: 'bg-green-500 text-white',
  error: 'bg-red-500 text-white',
  warning: 'bg-yellow-500 text-black',
  info: 'bg-blue-500 text-white',
};

const toastIcons = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

export function Toast({ toast, onDismiss }: ToastProps) {
  const [isExiting, setIsExiting] = useState(false);
  const Icon = toastIcons[toast.type];

  const handleDismiss = () => {
    setIsExiting(true);
    setTimeout(() => onDismiss(toast.id), 300); // Match animation duration
  };

  useEffect(() => {
    // Auto-dismiss after duration
    if (toast.duration > 0) {
      const timer = setTimeout(() => {
        handleDismiss();
      }, toast.duration);

      return () => clearTimeout(timer);
    }
  }, [toast.duration, toast.id]);

  return (
    <div
      className={`
        ${toastStyles[toast.type]}
        rounded-lg shadow-lg p-4 min-w-[300px] max-w-md
        flex items-start gap-3
        transform transition-all duration-300 ease-in-out
        ${isExiting ? 'translate-x-full opacity-0' : 'translate-x-0 opacity-100'}
      `}
    >
      <Icon className="w-5 h-5 flex-shrink-0 mt-0.5" />
      <p className="flex-1 text-sm font-medium">{toast.message}</p>
      <button
        onClick={handleDismiss}
        className="flex-shrink-0 hover:opacity-70 transition-opacity"
        aria-label="Dismiss notification"
      >
        <X className="w-5 h-5" />
      </button>
    </div>
  );
}
