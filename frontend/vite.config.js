import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
export default defineConfig({
  plugins: [vue()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    // 后端只绑 127.0.0.1 且无 CORS → 前端须同源经此转发 /api（changeOrigin:false 保留 Host，本地后端不按 Host 路由）
    proxy: { '/api': { target: 'http://127.0.0.1:8000', changeOrigin: false } },
  },
})
