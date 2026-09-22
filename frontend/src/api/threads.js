// 会话历史接口：列出会话 / 读取历史 / 删除会话。
// 后端 memory 未启用时返回 503。

export class ThreadsHttpError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`)
    this.name = 'ThreadsHttpError'
    this.status = status
    this.detail = detail
  }
}

async function jsonOrThrow(resp) {
  if (!resp.ok) {
    let detail = ''
    try {
      detail = (await resp.json()).detail || ''
    } catch {
      /* 非 JSON body 忽略 */
    }
    throw new ThreadsHttpError(resp.status, detail)
  }
  return resp.json()
}

export async function listThreads() {
  const data = await jsonOrThrow(await fetch('/api/threads'))
  return Array.isArray(data.items) ? data.items : []
}

export async function fetchThreadMessages(threadId) {
  const path = `/api/threads/${encodeURIComponent(threadId)}/messages`
  const data = await jsonOrThrow(await fetch(path))
  return Array.isArray(data.items) ? data.items : []
}

export async function deleteThread(threadId) {
  const path = `/api/threads/${encodeURIComponent(threadId)}`
  return jsonOrThrow(await fetch(path, { method: 'DELETE' }))
}
