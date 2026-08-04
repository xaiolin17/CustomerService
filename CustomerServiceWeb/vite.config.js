import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig(({ mode }) => {
  // 加载环境变量
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [vue()],
    server: {
      port: 3000,
      proxy: {
        // WebSocket 代理到后端
        '/ws': {
          target: env.VITE_WS_URL?.replace('ws://', 'http://') || 'http://localhost:8000',
          ws: true,
          changeOrigin: true,
        },
      },
    },
  }
})