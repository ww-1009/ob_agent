import { describe, expect, it } from 'vitest'
import { createSseDecoder, parseFrame } from '../src/lib/sse.js'

describe('parseFrame', () => {
  it('解析单条 data 帧（剥掉 data: 后的一个空格）', () => {
    expect(parseFrame('data: {"type":"done"}')).toEqual({ type: 'done' })
  })
  it('多行 data: 以换行拼接为同一 JSON 载荷', () => {
    expect(parseFrame('data: {"a":1,\ndata: "b":2}')).toEqual({ a: 1, b: 2 })
  })
  it('忽略 event/id/注释行，只取 data', () => {
    const raw = 'id: 1\nevent: msg\n: hi\ndata: {"type":"delta","text":"好"}'
    expect(parseFrame(raw)).toEqual({ type: 'delta', text: '好' })
  })
  it('容忍 CRLF', () => {
    expect(parseFrame('data: {"type":"done"}\r\n\r\n'.slice(0, -2))).toEqual({ type: 'done' })
  })
  it('无 data 行返回 null', () => {
    expect(parseFrame(': comment only')).toBeNull()
  })
  it('JSON 解析失败返回 null（不抛）', () => {
    expect(parseFrame('data: not-json')).toBeNull()
  })
})

describe('createSseDecoder', () => {
  it('按 \\n\\n 切帧并返回事件数组', () => {
    const push = createSseDecoder()
    expect(push('data: {"type":"status","text":"a"}\n\ndata: {"type":"done"}\n\n')).toEqual([
      { type: 'status', text: 'a' },
      { type: 'done' },
    ])
  })
  it('帧跨多次 push 也能正确切分（流式 chunk 边界）', () => {
    const push = createSseDecoder()
    const first = push('data: {"type":"de')
    expect(first).toEqual([])
    const rest = push('lta","text":"中"}\n\ndata: {"type":"done"}\n\n')
    expect(rest).toEqual([{ type: 'delta', text: '中' }, { type: 'done' }])
  })
  it('非 JSON 帧被跳过不影响后续', () => {
    const push = createSseDecoder()
    expect(push('data: oops\n\ndata: {"type":"done"}\n\n')).toEqual([{ type: 'done' }])
  })
})
