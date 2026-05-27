/* ESLint config for the Vite + React + TypeScript frontend.
 * Type-level checks (unused vars, undefined names, types) are owned by `tsc`
 * (`tsc --noEmit` / `npm run build`); ESLint here focuses on React-hooks
 * correctness and a light recommended baseline. */
module.exports = {
  root: true,
  env: { browser: true, es2021: true, node: true },
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 'latest',
    sourceType: 'module',
    ecmaFeatures: { jsx: true },
  },
  plugins: ['react-hooks'],
  extends: ['eslint:recommended', 'plugin:react-hooks/recommended'],
  ignorePatterns: [
    'dist',
    'node_modules',
    'coverage',
    '*.config.js',
    '*.config.ts',
    '*.config.cjs',
  ],
  rules: {
    // Owned by tsc — the base rules misfire on TS-only syntax (types/enums).
    'no-unused-vars': 'off',
    'no-undef': 'off',
  },
};
