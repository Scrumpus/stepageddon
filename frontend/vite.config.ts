import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // Single source of truth for the generation timing offset: read it from the
  // backend's .env so the receptor pulse stays aligned with the offset baked
  // into ML-generated note times. Falls back to the backend default (-30) when
  // the var is unset there. Resolved at build time and injected via `define`.
  const backendEnv = loadEnv(mode, path.resolve(__dirname, '../backend'), '')
  const generationTimingOffsetMs = Number(
    backendEnv.GENERATION_TIMING_OFFSET_MS ?? -30
  )

  // Fail the production build instead of silently shipping with the
  // localhost fallback baked in. Vite resolves import.meta.env at build
  // time, so a missing VITE_API_URL here = a broken prod bundle.
  if (mode === 'production' && !env.VITE_API_URL) {
    throw new Error(
      'VITE_API_URL must be set for production builds. ' +
      'Set it in frontend/.env or pass --build-arg VITE_API_URL=... to docker build.'
    )
  }

  return {
    define: {
      __GENERATION_TIMING_OFFSET_MS__: JSON.stringify(generationTimingOffsetMs),
    },
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, './src'),
        '@/types': path.resolve(__dirname, './src/types'),
        '@/features': path.resolve(__dirname, './src/features'),
        '@/components': path.resolve(__dirname, './src/components'),
        '@/hooks': path.resolve(__dirname, './src/hooks'),
        '@/utils': path.resolve(__dirname, './src/utils'),
        '@/config': path.resolve(__dirname, './src/config'),
        '@/lib': path.resolve(__dirname, './src/lib'),
        '@/app': path.resolve(__dirname, './src/app'),
      }
    },
    server: {
      port: 3000,
      proxy: {
        '/api': {
          target: 'http://localhost:8000',
          changeOrigin: true,
        }
      }
    }
  }
})
