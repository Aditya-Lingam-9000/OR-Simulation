/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        'or-bg': '#1a1a2e',
        'or-panel': '#16213e',
        'or-accent': '#0f3460',
        'or-highlight': '#e94560',
        'machine-on': '#22c55e',
        'machine-off': '#6b7280',
        'machine-standby': '#eab308',
        'machine-caution': '#ef4444',
        'source-rule': '#3b82f6',
        'source-llm': '#8b5cf6',
        'source-both': '#06b6d4',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'blink': 'blink 1.5s ease-in-out infinite',
        'slide-in': 'slideIn 0.3s ease-out',
      },
      keyframes: {
        blink: {
          '0%, 100%': { opacity: 1 },
          '50%': { opacity: 0.3 },
        },
        slideIn: {
          '0%': { transform: 'translateY(-10px)', opacity: 0 },
          '100%': { transform: 'translateY(0)', opacity: 1 },
        },
      },
    },
  },
  plugins: [],
};
