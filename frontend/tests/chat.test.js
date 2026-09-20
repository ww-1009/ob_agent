import { afterEach, describe, expect, it, vi } from 'vitest'
import { TextEncoder } from 'node:util'
import { ChatHttpError, postChat } from '../src/api/chat.js'

// 把字符串块包装成可被 TextDecoder 消费的 Uint8Array 流 reader
function streamFromChunks(chunks) {
  const bytes = chunks.map((c) => new TextEncoder().encode(c))
  let i = 0
  const reader = {
    released: false,
    async read() {
      if (i < bytes.length) {
        const value = bytes[i++]
        return { done: false, value }
      }
      return { done: true, value: undefined }
    },
    releaseLock() {
      this.released = true
    },
  }
  return reader
}

describe('postChat', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('200 流按序分发事件到 onEvent', async () => {
    const events = []
    const reader = streamFromChunks([
      'data: {"type":"status","text":"a"}\n\n',
      'data: {"type":"delta","text":"hi"}\n\ndata: {"type":"done"}\n\n',
    ])
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, body: { getReader: () => reader } })))
    await postChat({ messages: [{ role: 'user', content: 'q' }], onEvent: (ev) => events.push(ev) })
    expect(events).toEqual([
      { type: 'status', text: 'a' },
      { type: 'delta', text: 'hi' },
      { type: 'done' },
    ])
    expect(reader.released).toBe(true)
  })

  it('无 body（如 204）以 ChatHttpError reject', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 204, body: null })))
    await expect(postChat({ messages: [], onEvent() {} }))
      .rejects.toMatchObject({ name: 'ChatHttpError', status: 204, detail: 'empty response body' })
  })

  it('503 带 detail 时以 ChatHttpError(status=503) reject', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      status: 503,
      json: async () => ({ detail: 'LLM 未配置：请配置 backend/config.yaml 的 llm.*' }),
    })))
    await expect(postChat({ messages: [], onEvent() {} }))
      .rejects.toMatchObject({ name: 'ChatHttpError', status: 503, detail: 'LLM 未配置：请配置 backend/config.yaml 的 llm.*' })
  })

  it('错误体非 JSON 时以泛化 HTTP <status> message reject', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      status: 500,
      json: async () => { throw new Error('not json') },
    })))
    const err = await postChat({ messages: [], onEvent() {} }).catch((e) => e)
    expect(err).toBeInstanceOf(ChatHttpError)
    expect(err.status).toBe(500)
    expect(err.message).toBe('HTTP 500')
  })

  it('流读取异常时 finally 释放 reader', async () => {
    const reader = streamFromChunks(['data: {"type":"done"}\n\n'])
    reader.read = async () => { throw new Error('boom') }
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, status: 200, body: { getReader: () => reader } })))
    await expect(postChat({ messages: [], onEvent() {} })).rejects.toThrow('boom')
    expect(reader.released).toBe(true)
  })
})
