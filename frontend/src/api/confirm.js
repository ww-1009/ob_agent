export class ConfirmHttpError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`)
    this.name = 'ConfirmHttpError'
    this.status = status
    this.detail = detail
  }
}

// 把前端用户的「允许/拒绝」投递给后端 ConfirmationBroker 中挂起的确认。
// approved: boolean；signal 可随聊天流的中止一起取消。
export async function postConfirm({ requestId, approved, signal }) {
  const resp = await fetch('/api/chat/confirm', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ request_id: requestId, approved }),
    signal,
  })
  if (!resp.ok) {
    let detail = ''
    try {
      detail = (await resp.json()).detail || ''
    } catch { /* 非 JSON body 忽略 */ }
    throw new ConfirmHttpError(resp.status, detail)
  }
  return await resp.json()
}
