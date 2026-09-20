import { computed, reactive, ref } from 'vue'
import { ChatHttpError, postChat } from '../api/chat.js'
import { ConfirmHttpError, postConfirm } from '../api/confirm.js'
import { buildHistory, dedupeStatus } from './chatCore.js'

let seq = 0
const nextId = () => `m${++seq}`

export function useChat() {
  const messages = ref([]) // {id, role, content, status[], state:'streaming'|'done'|'error', error}
  const busy = ref(false)
  const llmNotConfigured = ref(false)
  // 待审批请求：{request_id, tool, tool_label, args, expires_in, state:'idle'|'deciding', error}
  const pendingConfirm = ref(null)
  let controller = null
  let sendSeq = 0 // latest-wins 令牌：与 busy/controller 同实例作用域；super-send 后旧流 finally 不误清新流共享态

  function mkAssistant() {
    return { id: nextId(), role: 'assistant', content: '', status: [], state: 'streaming', error: '' }
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
    const history = buildHistory(messages.value)
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
        messages: history,
        signal: controller.signal,
        onEvent: (ev) => {
          if (ev.type === 'status') {
            assistant.status = dedupeStatus(assistant.status, String(ev.text ?? ''))
          } else if (ev.type === 'delta') {
            pending += String(ev.text ?? '')
            scheduleFlush()
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
    }
  }

  async function decideConfirm(approved) {
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
      if (err instanceof ConfirmHttpError && (err.status === 404 || err.status === 409 || err.status === 503)) {
        // 请求已处理/不存在/通道不可用：不阻塞，关闭卡片（后端已按拒绝或已答复继续）
        pendingConfirm.value = null
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
    stop()
    messages.value = []
    llmNotConfigured.value = false
    pendingConfirm.value = null
  }

  const canSend = computed(() => !busy.value)

  return { messages, busy, canSend, llmNotConfigured, pendingConfirm, send, stop, clear, decideConfirm }
}
