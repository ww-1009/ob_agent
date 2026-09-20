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
// onEvent(ev) 收到 {type:'status'|'delta'|'error'|'done'}；须为同步回调以保事件顺序。AbortController 可中止。
export async function postChat({ messages, signal, onEvent }) {
  const resp = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ messages }),
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
