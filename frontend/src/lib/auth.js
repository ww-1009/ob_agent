// 访问控制：后端 auth.enabled=true 时，除 /api/health 外的接口都要求 Bearer 令牌。
// 采用「静态令牌 + 401 时弹一次性输入框」的最小方案，不引入登录页与用户体系。

const KEY = 'ob_agent.auth_token'

export function getToken() {
  try {
    return localStorage.getItem(KEY) || ''
  } catch {
    return '' // 隐私模式等禁用 localStorage
  }
}

export function setToken(token) {
  try {
    if (token) localStorage.setItem(KEY, token)
    else localStorage.removeItem(KEY)
  } catch {
    /* 忽略：持久化失败不影响本次会话 */
  }
}

function promptToken() {
  // 非浏览器环境（如单测）不弹窗
  if (typeof window === 'undefined' || typeof window.prompt !== 'function') return ''
  const t = window.prompt('后端已启用访问控制，请输入访问令牌（Bearer token）')
  return t ? t.trim() : ''
}

function withAuth(headers, token) {
  const h = { ...(headers || {}) }
  if (token) h.Authorization = `Bearer ${token}`
  return h
}

/** 统一带令牌请求；遇 401 时清除旧令牌、询问一次并重试一次。 */
export async function authedFetch(url, init = {}) {
  const token = getToken()
  const resp = await fetch(url, { ...init, headers: withAuth(init.headers, token) })
  if (resp.status !== 401) return resp

  setToken('') // 旧令牌已失效，避免每次请求都重复失败
  const asked = promptToken()
  if (!asked) return resp
  setToken(asked)
  return await fetch(url, { ...init, headers: withAuth(init.headers, asked) })
}
