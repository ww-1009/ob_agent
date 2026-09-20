// 聊天纯逻辑（与 Vue 解耦，便于单测）：历史裁剪 / status 去重。

export function buildHistory(messages) {
  const out = []
  for (const m of messages) {
    if (m.role === 'user') {
      out.push({ role: 'user', content: m.content })
    } else if (m.role === 'assistant' && m.state === 'done' && m.content) {
      out.push({ role: 'assistant', content: m.content })
    }
  }
  return out
}

export function dedupeStatus(list, text) {
  return list[list.length - 1] === text ? list : [...list, text]
}
