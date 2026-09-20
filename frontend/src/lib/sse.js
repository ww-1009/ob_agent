// SSE data-only 帧解析。纯函数/闭包，浏览器与测试环境通用。
// 后端帧形态：`data: {json}\n\n`（单 data 行）；这里额外兼容多 data 行 / 注释 / event 行 / CRLF。

export function parseFrame(raw) {
  const dataLines = []
  for (const line of raw.split('\n')) {
    const l = line.endsWith('\r') ? line.slice(0, -1) : line
    if (l.startsWith('data:')) dataLines.push(l.slice(5).replace(/^ /, ''))
    // event/id/retry/注释(: 开头) 一律忽略
  }
  if (dataLines.length === 0) return null
  try {
    return JSON.parse(dataLines.join('\n'))
  } catch {
    return null
  }
}

export function createSseDecoder() {
  let buffer = ''
  return function push(chunk) {
    buffer += chunk
    const events = []
    let idx
    while ((idx = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, idx)
      buffer = buffer.slice(idx + 2)
      const ev = parseFrame(raw)
      if (ev !== null) events.push(ev)
    }
    return events
  }
}
