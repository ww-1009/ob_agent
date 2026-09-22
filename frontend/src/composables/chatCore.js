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

// 后端持久化历史（{role, content}）→ UI 消息对象；字段需与 MessageBubble.vue 期望一致。
// 持久化历史只有终稿，没有流式状态/工具灰字，因此状态恒为 done。
export function toMessageView(items) {
  if (!Array.isArray(items)) return []
  return items
    .filter((m) => m && (m.role === 'user' || m.role === 'assistant'))
    .map((m) => ({
      role: m.role,
      content: String(m.content ?? ''),
      status: [],
      state: 'done',
      error: '',
    }))
}

export function threadLabel(thread) {
  const s = thread && typeof thread.title === 'string' ? thread.title.trim() : ''
  return s || '新对话'
}

