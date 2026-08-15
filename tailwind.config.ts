import type { Config } from 'tailwindcss';

const config: Config = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          blue: {
            50: '#F4FAFE',
            100: '#E6F4FB',
            200: '#D0ECF8',
            300: '#BFE3F7',
            400: '#94D0F3',
            500: '#A8D8F0',
            600: '#5CB4E8',
            700: '#2A92D7',
            800: '#1D6EB0',
            900: '#174F81',
          },
          accent: '#A8D8F0',
          glow: 'rgba(168, 216, 240, 0.35)',
        },
        ink: {
          50: '#F7F8FA',
          100: '#EFEFEF',
          200: '#E2E4E8',
          300: '#C7CAD0',
          400: '#8E949F',
          500: '#5C6370',
          700: '#2A2E35',
          800: '#16181C',
          900: '#0A0A0A',
          950: '#050505',
        },
        verdict: {
          genuine: '#10B981',
          suspicious: '#F59E0B',
          manipulated: '#EF4444',
        },
      },
      borderRadius: {
        '2xl': '16px',
        '3xl': '24px',
        '4xl': '32px',
      },
      boxShadow: {
        'glow-sm': '0 0 15px rgba(168, 216, 240, 0.25)',
        'glow-md': '0 0 30px rgba(168, 216, 240, 0.35)',
        'glow-lg': '0 0 50px rgba(168, 216, 240, 0.45)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.06)',
        'dark-glass': '0 8px 32px 0 rgba(0, 0, 0, 0.5)',
      },
      animation: {
        'pulse-slow': 'pulse 4s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'float': 'float 6s ease-in-out infinite',
        'glitch': 'glitch 2s infinite',
      },
      keyframes: {
        float: {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-10px)' },
        },
        glitch: {
          '0%, 100%': { transform: 'translate(0)' },
          '20%': { transform: 'translate(-2px, 2px)' },
          '40%': { transform: 'translate(-2px, -2px)' },
          '60%': { transform: 'translate(2px, 2px)' },
          '80%': { transform: 'translate(2px, -2px)' },
        },
      },
    },
  },
  plugins: [],
};

export default config;
