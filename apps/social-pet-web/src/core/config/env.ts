export const appEnv = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:3001',
  devTools: (import.meta.env.VITE_DEV_TOOLS ?? '').toLowerCase() === 'true'
};
