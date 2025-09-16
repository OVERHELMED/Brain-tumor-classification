/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        // Tony Stark color palette
        'stark': {
          'black': '#0a0a0a',
          'navy': '#1a1a2e',
          'blue': '#16213e',
          'cyan': '#00d4ff',
          'electric': '#00ffff',
          'neon': '#00ff88',
          'amber': '#ffaa00',
          'red': '#ff3366',
          'purple': '#8b5cf6',
          'gray': {
            900: '#0f0f23',
            800: '#1a1a2e',
            700: '#16213e',
            600: '#233876',
            500: '#4a5568',
            400: '#718096',
            300: '#a0aec0',
            200: '#e2e8f0',
            100: '#f7fafc',
          }
        }
      },
      fontFamily: {
        'mono': ['JetBrains Mono', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        'sans': ['Inter', 'system-ui', 'sans-serif'],
      },
      animation: {
        'glow': 'glow 2s ease-in-out infinite alternate',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'scan': 'scan 2s linear infinite',
        'float': 'float 6s ease-in-out infinite',
        'matrix': 'matrix 20s linear infinite',
        'hologram': 'hologram 4s ease-in-out infinite',
      },
      keyframes: {
        glow: {
          '0%': { 
            'box-shadow': '0 0 5px #00d4ff, 0 0 10px #00d4ff, 0 0 15px #00d4ff',
            'border-color': '#00d4ff'
          },
          '100%': { 
            'box-shadow': '0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff',
            'border-color': '#00ffff'
          }
        },
        scan: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(100%)' }
        },
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' }
        },
        matrix: {
          '0%': { transform: 'translateY(-100%)' },
          '100%': { transform: 'translateY(100vh)' }
        },
        hologram: {
          '0%, 100%': { opacity: '0.8', transform: 'scale(1)' },
          '50%': { opacity: '1', transform: 'scale(1.02)' }
        }
      },
      backdropBlur: {
        'xs': '2px',
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
        'circuit': "url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAiIGhlaWdodD0iNDAiIHZpZXdCb3g9IjAgMCA0MCA0MCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTIwIDIwSDQwTTIwIDIwVjBNMjAgMjBWNDBNMjAgMjBIMCIgc3Ryb2tlPSIjMDBkNGZmIiBzdHJva2Utd2lkdGg9IjAuNSIgb3BhY2l0eT0iMC4xIi8+Cjwvc3ZnPgo=')",
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
