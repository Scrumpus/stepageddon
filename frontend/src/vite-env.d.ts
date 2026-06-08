/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_URL?: string;
  // add more env variables as needed
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

// Injected at build time by vite.config.ts from the backend's
// GENERATION_TIMING_OFFSET_MS (single source of truth).
declare const __GENERATION_TIMING_OFFSET_MS__: number;
