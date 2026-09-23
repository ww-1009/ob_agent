import { computed, reactive, ref } from 'vue'
import { ChatHttpError, postChat } from '../api/chat.js'
import { ConfirmHttpError, postConfirm } from '../api/confirm.js'
import { fetchHealth } from '../api/health.js'
import { deleteThread, fetchThreadMessages, listThreads, ThreadsHttpError } from '../api/threads.js'
import { AuditHttpError, listAudit } from '../api/audit.js'
import { buildHistory, dedupeStatus, toMessageView } from './chatCore.js'
import { loadThreadId, newThreadId, saveThreadId } from '../lib/threadId.js'

let seq = 0
const nextId = () => `m${++seq}`

export function useChat() {
  const messages = ref([]) // {id, role, content, status[], state:'streaming'|'done'|'error', error}
  const busy = ref(false)
  const llmNotConfigured = ref(false)
  // 待审批请求：{request_id, tool, tool_label, args, expires_in, state:'idle'|'deciding', error}
  const pendingConfirm = ref(null)
  // 会话记忆（后端 PG 持久化）：memoryEnabled 为 false 时退回「前端持有全量历史」的无状态模式
  const memoryEnabled = ref(false)
  const threadId = ref('')
  const threads = ref([]) // [{thread_id, title, created_at, updated_at, message_count}]
  const loadingThread = ref(false)
  // 工具审计（后端 audit_event 表）：工具轨迹的持久化留痕
  const auditEvents = ref([])
  const loadingAudit = ref(false)
  let controller = null
  let sendSeq = 0 // latest-wins 令牌：与 busy/controller 同实例作用域；super-send 后旧流 finally 不误清新流共享态

  function mkAssistant() {
    return { id: nextId(), role: 'assistant', content: '', status: [], tools: [], state: 'streaming', error: '' }
  }

  function applyHistory(items) {
    messages.value = toMessageView(items).map((m) => ({ id: nextId(), ...m }))
  }

  /** 会话列表：503 说明服务端记忆关闭，顺势退回无状态模式。 */
  async function refreshThreads() {
    if (!memoryEnabled.value) return
    try {
      threads.value = await listThreads()
    } catch (err) {
      if (err instanceof ThreadsHttpError && err.status === 503) memoryEnabled.value = false
    }
  }

  /** 工具审计：默认查当前会话；threadIdFilter 传空则查全部会话。 */
  async function refreshAudit(threadIdFilter) {
    if (!memoryEnabled.value) return
    loadingAudit.value = true
    try {
      const tid = threadIdFilter === undefined ? threadId.value : threadIdFilter
      auditEvents.value = await listAudit(tid ? { threadId: tid } : {})
    } catch (err) {
      if (err instanceof AuditHttpError && err.status === 503) memoryEnabled.value = false
    } finally {
      loadingAudit.value = false
    }
  }

  async function loadThreadMessages(id) {
    if (!id) return
    loadingThread.value = true
    try {
      applyHistory(await fetchThreadMessages(id))
    } catch (err) {
      // 历史加载失败不阻塞对话：保持当前视图（本地记录的会话可能已被删除）
      if (err instanceof ThreadsHttpError && err.status === 503) memoryEnabled.value = false
    } finally {
      loadingThread.value = false
    }
  }

  /** 启动时对齐服务端能力：记忆开启则恢复上次会话（localStorage）与列表。 */
  async function init() {
    const health = await fetchHealth()
    memoryEnabled.value = !!(health && health.memory_enabled)
    if (!memoryEnabled.value) return
    threadId.value = loadThreadId() || newThreadId()
    saveThreadId(threadId.value)
    await Promise.all([refreshThreads(), loadThreadMessages(threadId.value)])
  }

  async function openThread(id) {
    if (!id) return
    stop()
    pendingConfirm.value = null
    llmNotConfigured.value = false
    threadId.value = id
    saveThreadId(id)
    messages.value = []
    await loadThreadMessages(id)
  }

  function newThread() {
    stop()
    messages.value = []
    pendingConfirm.value = null
    llmNotConfigured.value = false
    if (memoryEnabled.value) {
      threadId.value = newThreadId()
      saveThreadId(threadId.value)
    }
  }

  async function removeThread(id) {
    if (!id) return
    try {
      await deleteThread(id)
    } catch {
      /* 删除失败：保持列表原状，仅刷新 */
    }
    if (id === threadId.value) newThread()
    await refreshThreads()
  }

  async function send(text) {
    const q = String(text).trim()
    if (!q) return
    stop() // 双保险：先中止可能在途的流
    const userMsg = { id: nextId(), role: 'user', content: q, status: [], state: 'done', error: '' }
    const assistant = reactive(mkAssistant()) // reactive：流式 content/status/state 写走 Proxy，触发组件更新
    messages.value.push(userMsg, assistant)
    busy.value = true
    llmNotConfigured.value = false
    pendingConfirm.value = null
    controller = new AbortController()
    const mySend = ++sendSeq // 本流令牌：finally 里只有最新 send 才允许复位共享态
    // 无记忆：需回传完整历史；有记忆：只传本轮消息，历史由服务端检查点提供
    const useMemory = memoryEnabled.value && !!threadId.value
    const payload = useMemory
      ? [{ role: 'user', content: q }]
      : buildHistory(messages.value)
    // rAF 节流：同一帧内多个 delta 合并为一次 content 变更，避免逐 chunk 重渲染 markdown
    let rafId = 0
    let pending = ''
    const flushPending = () => {
      rafId = 0
      if (pending) {
        assistant.content += pending
        pending = ''
      }
    }
    const scheduleFlush = () => {
      if (!rafId) rafId = requestAnimationFrame(flushPending)
    }
    try {
      await postChat({
        messages: payload,
        threadId: useMemory ? threadId.value : undefined,
        signal: controller.signal,
        onEvent: (ev) => {
          if (ev.type === 'status') {
            assistant.status = dedupeStatus(assistant.status, String(ev.text ?? ''))
          } else if (ev.type === 'delta') {
            pending += String(ev.text ?? '')
            scheduleFlush()
          } else if (ev.type === 'tool') {
            // 工具轨迹：只含入参摘要/耗时/行数/成败，不含行数据
            assistant.tools = [...assistant.tools, {
              id: String(ev.id ?? ''),
              name: String(ev.name ?? ''),
              label: String(ev.label ?? ev.name ?? ''),
              args: (ev.args && typeof ev.args === 'object') ? { ...ev.args } : {},
              ok: ev.ok !== false,
              error: ev.error ? String(ev.error) : '',
              rows: Number.isInteger(ev.rows) ? ev.rows : null,
              truncated: ev.truncated === true,
              approved: (ev.approved === true || ev.approved === false) ? ev.approved : null,
              duration_ms: Number.isFinite(ev.duration_ms) ? ev.duration_ms : null,
              // 只有 get_sql_explain 会带归一化计划；其余工具为 null，ToolTrace 不渲染计划块
              plan: (ev.plan && typeof ev.plan === 'object') ? ev.plan : null,
            }]
          } else if (ev.type === 'confirm_request') {
            pendingConfirm.value = {
              request_id: String(ev.request_id ?? ''),
              tool: String(ev.tool ?? ''),
              tool_label: String(ev.tool_label ?? ev.tool ?? ''),
              args: (ev.args && typeof ev.args === 'object') ? { ...ev.args } : {},
              expires_in: Number(ev.expires_in) > 0 ? Number(ev.expires_in) : 120,
              requested_at: Date.now(),
              state: 'idle',
              error: '',
            }
          } else if (ev.type === 'confirm_timeout') {
            const cur = pendingConfirm.value
            if (cur && String(ev.request_id ?? '') === cur.request_id) {
              cur.error = '审批超时，已默认拒绝'
              pendingConfirm.value = null
            }
          } else if (ev.type === 'error') {
            flushPending()
            assistant.error = String(ev.message ?? '未知错误')
            assistant.state = 'error'
          } else if (ev.type === 'done') {
            flushPending()
            assistant.state = 'done'
          }
        },
      })
    } catch (err) {
      if (err && err.name === 'AbortError') {
        assistant.state = 'done' // 用户停止：保留已收文本，不算错误
      } else if (err instanceof ChatHttpError && err.status === 503) {
        llmNotConfigured.value = true
        assistant.error = err.detail || 'LLM 未配置'
        assistant.state = 'error'
      } else {
        assistant.error = (err && err.message) ? err.message : String(err)
        assistant.state = 'error'
      }
    } finally {
      if (rafId) cancelAnimationFrame(rafId)
      flushPending() // 中止/异常路径也确保尾部文本不丢
      if (assistant.state === 'streaming') assistant.state = 'done' // 防御：干净 EOF 无 done 终帧时不永久卡 streaming
      pendingConfirm.value = null // 流已结束：清掉所有残留待审批卡片
      if (mySend === sendSeq) {
        busy.value = false
        controller = null
      }
      // 后端在 done 之前已落库/落审计，这里刷新即可看到新会话与新轨迹
      if (useMemory) {
        void refreshThreads()
        void refreshAudit()
      }
    }
  }

  async function decideConfirm(approved, reason = '') {
    const confirm = pendingConfirm.value
    if (!confirm || confirm.state === 'deciding') return
    if (confirm.state === 'resolved') {
      pendingConfirm.value = null
      return
    }
    confirm.state = 'deciding'
    try {
      await postConfirm({
        requestId: confirm.request_id,
        approved,
        signal: controller ? controller.signal : undefined,
      })
      pendingConfirm.value = null // 已投递，后端继续处理，流会随之恢复
    } catch (err) {
      if (err && err.name === 'AbortError') {
        // 流被中止，审批作废；finally 会复位共享态
        pendingConfirm.value = null
        return
      }
      if (err instanceof ConfirmHttpError && err.status === 409) {
        // 已答复/已被处理：幂等，安静关闭
        pendingConfirm.value = null
        return
      }
      if (err instanceof ConfirmHttpError && (err.status === 404 || err.status === 503)) {
        if (reason === 'timeout') {
          // 本地倒计时到期的尽力投递：服务端通常已按超时拒绝，静默收尾
          pendingConfirm.value = null
          return
        }
        // 不能再假装"已答复"：后端并未接受这次审批，把原因留在卡片上让用户可见
        confirm.error = err.detail
          || (err.status === 404 ? '确认请求已失效（可能已被处理或已过期）' : '确认通道不可用，请重试')
        confirm.state = 'idle'
        return
      }
      confirm.error = (err && err.message) ? err.message : String(err)
      confirm.state = 'idle' // 允许重试
    }
  }

  function stop() {
    if (controller) {
      controller.abort()
      controller = null
    }
  }

  function clear() {
    newThread()
  }

  const canSend = computed(() => !busy.value)

  return {
    messages, busy, canSend, llmNotConfigured, pendingConfirm,
    memoryEnabled, threadId, threads, loadingThread,
    auditEvents, loadingAudit,
    send, stop, clear, decideConfirm,
    init, openThread, newThread, removeThread, refreshThreads, refreshAudit,
  }
}
