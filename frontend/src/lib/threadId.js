// 会话 ID：由前端生成（后端检查点与 chat_message 都用它作 thread_id）。
// 需匹配后端白名单 ^[A-Za-z0-9_-]{1,64}$。

const KEY = 'ob_agent.thread_id'

export function newThreadId() {
  const c = globalThis.crypto
  if (c && typeof c.randomUUID === 'function') return c.randomUUID()
  // 退化路径（无 crypto.randomUUID 的环境）：仍满足白名单
  return `t-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

export function loadThreadId() {
  try {
    return localStorage.getItem(KEY) || ''
  } catch {
    return '' // 隐私模式等禁用 localStorage
  }
}

export function saveThreadId(id) {
  try {
    localStorage.setItem(KEY, String(id || ''))
  } catch {
    /* 忽略：持久化失败不影响本次会话 */
  }
}
