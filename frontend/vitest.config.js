import { defineConfig } from 'vitest/config'
export default defineConfig({
  test: {
    // DOMPurify(markdown 消毒) 需要 window，故用 jsdom 环境
    environment: 'jsdom',
    include: ['tests/**/*.test.js'],
  },
})
