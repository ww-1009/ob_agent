// 审计接口：查询只读工具调用留痕（后端 memory 未启用时返回 503）。

import { authedFetch } from '../lib/auth.js'

export class AuditHttpError extends Error {
  constructor(status, detail) {
    super(detail || `HTTP ${status}`)
    this.name = 'AuditHttpError'
    this.status = status
    this.detail = detail
  }
}

export async function listAudit({ threadId, tool, limit } = {}) {
  const q = new URLSearchParams()
  if (threadId) q.set('thread_id', threadId)
  if (tool) q.set('tool', tool)
  if (limit) q.set('limit', String(limit))
  const suffix = q.toString() ? `?${q.toString()}` : ''
  const resp = await authedFetch(`/api/audit${suffix}`)
  if (!resp.ok) {
    let detail = ''
    try {
      detail = (await resp.json()).detail || ''
    } catch {
      /* 非 JSON body 忽略 */
    }
    throw new AuditHttpError(resp.status, detail)
  }
  const data = await resp.json()
  return Array.isArray(data.items) ? data.items : []
}
