import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, watchEffect } from 'vue'
import { ChatHttpError, postChat } from '../src/api/chat.js'
import { ConfirmHttpError, postConfirm } from '../src/api/confirm.js'
import { useChat } from '../src/composables/useChat.js'

// mock 掉 postChat：保留真实 ChatHttpError（测 503 instanceof），postChat 换成可控 deferred。
vi.mock('../src/api/chat.js', async (importOriginal) => {
  const actual = await importOriginal()
  return { ...actual, postChat: vi.fn() }
})

// mock 掉 postConfirm：审批投递失败路径需要逐用例指定状态码
vi.mock('../src/api/confirm.js', async (importOriginal) => {
  const actual = await importOriginal()
  return { ...actual, postConfirm: vi.fn() }
})

// 每个 send 调用记录一条可控 rec；resolve/reject 由用例手动触发以模拟流结束。
let recs = []

function plainMsgs(chat) {
  return JSON.parse(JSON.stringify(chat.messages.value))
}

const abortErr = () => Object.assign(new Error('aborted'), { name: 'AbortError' })

describe('useChat', () => {
  beforeEach(() => {
    recs = []
    vi.mocked(postChat).mockClear()
    vi.mocked(postConfirm).mockClear()
    // deferred：不自动完成，测试自行 rec.resolve()/reject()
    vi.mocked(postChat).mockImplementation(({ messages, signal, onEvent }) => {
      const rec = { messages, signal, onEvent, resolve: null, reject: null }
      rec.promise = new Promise((resolve, reject) => {
        rec.resolve = resolve
        rec.reject = reject
      })
      recs.push(rec)
      return rec.promise
    })
    // rAF 同步执行：delta 立即写入 content，便于断言；返回 0 使 flushPending 后 rafId 归零，每个 delta 各自触发
    vi.stubGlobal('requestAnimationFrame', (cb) => {
      cb()
      return 0
    })
    vi.stubGlobal('cancelAnimationFrame', () => {})
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.mocked(postChat).mockClear()
    vi.mocked(postConfirm).mockClear()
  })

  // 发起一轮对话并注入一条待审批卡片，返回清理用的收尾函数
  function withPendingConfirm() {
    const chat = useChat()
    const p = chat.send('执行一下')
    const rec = recs[0]
    rec.onEvent({
      type: 'confirm_request',
      request_id: 'cf_1',
      tool: 'execute_sql',
      tool_label: '执行只读 SQL 查询',
      args: { sql: 'select * from orders' },
      expires_in: 120,
    })
    const finish = async () => {
      rec.onEvent({ type: 'done' })
      rec.resolve()
      await p
    }
    return { chat, finish }
  }

  it('happy：status 去重 + delta 累积 + done 终态，且 history 只带已完成内容', async () => {
    const chat = useChat()
    const p = chat.send('你好')
    expect(recs).toHaveLength(1)
    const rec = recs[0]
    expect(chat.busy.value).toBe(true)
    expect(chat.canSend.value).toBe(false)
    // send 时 assistant 仍在 streaming → buildHistory 只该带 user
    expect(rec.messages).toEqual([{ role: 'user', content: '你好' }])

    rec.onEvent({ type: 'status', text: '思考中' })
    rec.onEvent({ type: 'status', text: '思考中' }) // 相邻重复应被去重
    rec.onEvent({ type: 'status', text: '整理答案' })
    rec.onEvent({ type: 'delta', text: '你好，' })
    rec.onEvent({ type: 'delta', text: 'OceanBase！' })
    rec.onEvent({ type: 'done' })
    rec.resolve()
    await p

    expect(chat.busy.value).toBe(false)
    expect(chat.canSend.value).toBe(true)
    expect(chat.llmNotConfigured.value).toBe(false)
    const msgs = plainMsgs(chat)
    expect(msgs).toHaveLength(2)
    expect(msgs[0]).toMatchObject({ role: 'user', content: '你好', state: 'done' })
    expect(msgs[1]).toMatchObject({
      role: 'assistant',
      content: '你好，OceanBase！',
      status: ['思考中', '整理答案'],
      state: 'done',
      error: '',
    })
  })

  it('流式增量走 reactive Proxy：effect 能观测到 content 逐段更新', async () => {
    const chat = useChat()
    const p = chat.send('hi')
    await nextTick() // 让 send 推送消息后的调度先 flush，再挂 effect 得到干净的基线
    const seen = []
    const stopEffect = watchEffect(() => {
      let last = null
      for (const m of chat.messages.value) {
        if (m.role === 'assistant') last = m.content
      }
      seen.push(last)
    })
    expect(seen).toEqual(['']) // 基线：assistant 内容为空

    const rec = recs[0]
    rec.onEvent({ type: 'delta', text: 'react' }) // rAF 同步 flush → content 写 'react'
    await nextTick()
    // 若 content 写的是 raw（绕过 Proxy），此 effect 不会重跑，seen 停留 ['']
    expect(seen).toEqual(['', 'react'])

    rec.onEvent({ type: 'done' })
    rec.resolve()
    await p
    expect(plainMsgs(chat)[1]).toMatchObject({ content: 'react', state: 'done' })
    stopEffect()
  })

  it('stop/AbortError：保留已收文本、终态 done 而非 error', async () => {
    const chat = useChat()
    const p = chat.send('q1')
    const rec = recs[0]
    rec.onEvent({ type: 'status', text: '准备' })
    rec.onEvent({ type: 'delta', text: '已收' })
    rec.reject(abortErr())
    await p

    expect(chat.busy.value).toBe(false)
    expect(chat.llmNotConfigured.value).toBe(false)
    expect(plainMsgs(chat)[1]).toMatchObject({
      content: '已收',
      status: ['准备'],
      state: 'done',
      error: '',
    })
  })

  it('HTTP 503：llmNotConfigured 置位、assistant error 有值、state error', async () => {
    const chat = useChat()
    const p = chat.send('q2')
    recs[0].reject(new ChatHttpError(503, 'LLM 未配置：请配置 backend/config.yaml 的 llm.*'))
    await p

    expect(chat.llmNotConfigured.value).toBe(true)
    expect(chat.busy.value).toBe(false)
    expect(plainMsgs(chat)[1]).toMatchObject({
      state: 'error',
      error: 'LLM 未配置：请配置 backend/config.yaml 的 llm.*',
    })
  })

  it('未知错误：state error、error 透传 err.message', async () => {
    const chat = useChat()
    const p = chat.send('q3')
    recs[0].reject(new Error('boom'))
    await p

    expect(chat.busy.value).toBe(false)
    expect(chat.llmNotConfigured.value).toBe(false)
    expect(plainMsgs(chat)[1]).toMatchObject({ state: 'error', error: 'boom' })
  })

  it('#2 回归：super-send 时旧流晚到的 finally 不误清新流 busy/controller', async () => {
    const chat = useChat()
    const pA = chat.send('A')
    const pB = chat.send('B')
    expect(recs).toHaveLength(2)
    const [recA, recB] = recs
    expect(chat.busy.value).toBe(true) // B 在途

    // A 先结束（其 finally 晚到于 B 已接管之后）：不得清 busy/controller
    recA.resolve()
    await pA
    expect(chat.busy.value).toBe(true) // ← 关键回归点：旧流 finally 不得误清新流

    // B 仍是当前流：stop() 应仍能中止 B
    chat.stop()
    recB.reject(abortErr())
    await pB
    expect(chat.busy.value).toBe(false)

    const msgs = plainMsgs(chat)
    expect(msgs).toHaveLength(4) // A 轮 + B 轮各 user+assistant
    expect(msgs[1]).toMatchObject({ role: 'assistant', state: 'done', content: '' }) // A 无事件 → 防御转 done
    expect(msgs[3]).toMatchObject({ role: 'assistant', state: 'done', content: '' }) // B 被 stop → done
  })

  it('503 后 clear()：清空消息并复位 llmNotConfigured', async () => {
    const chat = useChat()
    const p = chat.send('q4')
    recs[0].reject(new ChatHttpError(503, 'LLM 未配置'))
    await p
    expect(chat.llmNotConfigured.value).toBe(true)
    expect(chat.messages.value).toHaveLength(2)

    chat.clear()
    expect(chat.messages.value).toEqual([])
    expect(chat.llmNotConfigured.value).toBe(false)
    expect(chat.busy.value).toBe(false)
  })

  it('审批成功：卡片关闭', async () => {
    const { chat, finish } = withPendingConfirm()
    vi.mocked(postConfirm).mockResolvedValueOnce({ ok: true })
    await chat.decideConfirm(true)
    expect(chat.pendingConfirm.value).toBeNull()
    await finish()
  })

  it('审批 404：不再静默关闭卡片，错误可见且允许重试', async () => {
    const { chat, finish } = withPendingConfirm()
    vi.mocked(postConfirm).mockRejectedValueOnce(
      new ConfirmHttpError(404, '确认请求不存在或已过期'),
    )
    await chat.decideConfirm(true)
    expect(chat.pendingConfirm.value).not.toBeNull()
    expect(chat.pendingConfirm.value.error).toBe('确认请求不存在或已过期')
    expect(chat.pendingConfirm.value.state).toBe('idle') // 可重试
    await finish()
  })

  it('审批 503：卡片保留并提示通道不可用', async () => {
    const { chat, finish } = withPendingConfirm()
    vi.mocked(postConfirm).mockRejectedValueOnce(new ConfirmHttpError(503, ''))
    await chat.decideConfirm(true)
    expect(chat.pendingConfirm.value.error).toContain('确认通道不可用')
    expect(chat.pendingConfirm.value.state).toBe('idle')
    await finish()
  })

  it('审批 409：已答复/已处理，幂等关闭', async () => {
    const { chat, finish } = withPendingConfirm()
    vi.mocked(postConfirm).mockRejectedValueOnce(new ConfirmHttpError(409, ''))
    await chat.decideConfirm(true)
    expect(chat.pendingConfirm.value).toBeNull()
    await finish()
  })

  it('倒计时到期（reason=timeout）的 404：静默收尾，不显示错误', async () => {
    const { chat, finish } = withPendingConfirm()
    vi.mocked(postConfirm).mockRejectedValueOnce(new ConfirmHttpError(404, ''))
    await chat.decideConfirm(false, 'timeout')
    expect(chat.pendingConfirm.value).toBeNull()
    await finish()
  })
})
