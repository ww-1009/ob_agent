import { createSseDecoder } from '../lib/sse.js'

export class ChatHttpError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`)
    this.name = 'ChatHttpError'
    this.status = status
    this.detail = detail
  }
}

// messages: [{role:'user'|'assistant', content}]
// threadId: 启用会话记忆时传入；此时后端只把 messages 的最后一条当作本轮新消息
// （历史由服务端检查点提供），调用方应只传本轮用户消息。
// onEvent(ev) 收到 {type:'status'|'delta'|'error'|'done'}；须为同步回调以保事件顺序。AbortController 可中止。
export async function postChat({ messages, threadId, signal, onEvent }) {
  const body = threadId ? { thread_id: threadId, messages } : { messages }
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })
  if (!resp.ok) {
    let detail = ''
    try {
      detail = (await resp.json()).detail || ''
    } catch { /* 非 JSON body 忽略 */ }
    throw new ChatHttpError(resp.status, detail)
  }
  if (!resp.body) throw new ChatHttpError(resp.status, 'empty response body')
  const reader = resp.body.getReader()
  // createSseDecoder() 返回 push(chunk) 函数（非对象），直接调用
  const push = createSseDecoder()
  const text = new TextDecoder()
  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      const events = push(text.decode(value, { stream: true }))
      for (const ev of events) onEvent(ev)
    }
    // 流已关闭：冲刷 decoder 缓冲尾部 + TextDecoder 未完成码点，防丢尾帧
    const tail = push(text.decode())
    for (const ev of tail) onEvent(ev)
  } finally {
    try { reader.releaseLock() } catch { /* 已中断 */ }
  }
}
