import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@configs': path.resolve(__dirname, '..', 'configs'),
    },
  },
  server: {
    port: 3000,
    fs: {
      allow: ['..'],  // allow serving files from parent (configs/)
    },
    proxy: {
      '/health': 'http://127.0.0.1:8000',
      '/state': 'http://127.0.0.1:8000',
      '/stats': 'http://127.0.0.1:8000',
      '/surgeries': 'http://127.0.0.1:8000',
      '/machines': 'http://127.0.0.1:8000',
      '/select_surgery': 'http://127.0.0.1:8000',
      '/transcript': 'http://127.0.0.1:8000',
      '/override': 'http://127.0.0.1:8000',
      '/ws': {
        target: 'ws://127.0.0.1:8000',
        ws: true,
      },
    },
  },
});
