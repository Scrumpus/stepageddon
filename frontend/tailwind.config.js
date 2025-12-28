/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'game-bg': '#0f0f1e',
        'game-primary': '#6366f1',
        'game-secondary': '#8b5cf6',
        'game-accent': '#ec4899',
      },
      animation: {
        'arrow-scroll': 'scroll 2s linear',
        'pulse-hit': 'pulse-hit 0.3s ease-out',
      },
      keyframes: {
        scroll: {
          '0%': { transform: 'translateY(-100vh)' },
          '100%': { transform: 'translateY(0)' },
        },
        'pulse-hit': {
          '0%, 100%': { transform: 'scale(1)', opacity: '1' },
          '50%': { transform: 'scale(1.3)', opacity: '0.7' },
        },
      },
    },
  },
  plugins: [],
}
