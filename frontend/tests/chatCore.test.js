import { describe, expect, it } from 'vitest'
import { buildHistory, dedupeStatus, toolMeta } from '../src/composables/chatCore.js'

describe('buildHistory', () => {
  it('user 消息总带上', () => {
    const msgs = [{ role: 'user', content: 'hi', state: 'done' }]
    expect(buildHistory(msgs)).toEqual([{ role: 'user', content: 'hi' }])
  })
  it('只带已完成且非空的 assistant 终态', () => {
    const msgs = [
      { role: 'user', content: 'q', state: 'done' },
      { role: 'assistant', content: 'a1', state: 'done' },
      { role: 'assistant', content: '半截', state: 'streaming' },
      { role: 'assistant', content: '', state: 'done' },
      { role: 'assistant', content: 'err', state: 'error', error: 'boom' },
    ]
    expect(buildHistory(msgs)).toEqual([
      { role: 'user', content: 'q' },
      { role: 'assistant', content: 'a1' },
    ])
  })
})

describe('dedupeStatus', () => {
  it('相邻重复不追加', () => {
    expect(dedupeStatus(['a'], 'a')).toEqual(['a'])
    expect(dedupeStatus(['a'], 'b')).toEqual(['a', 'b'])
  })
})

describe('toolMeta', () => {
  it('行数/截断/耗时按需拼接', () => {
    expect(toolMeta({ rows: 2, truncated: true, duration_ms: 12 })).toBe('2 行 · 结果已截断 · 12 ms')
    expect(toolMeta({})).toBe('')
    expect(toolMeta(null)).toBe('')
  })
  it('执行计划轨迹在折叠状态下也显示算子数', () => {
    expect(toolMeta({ plan: { node_count: 3 } })).toBe('算子 3')
    expect(toolMeta({ rows: 0, duration_ms: 5, plan: { node_count: 3 } })).toBe('0 行 · 5 ms · 算子 3')
    expect(toolMeta({ plan: { nodes: [] } })).toBe('') // 缺 node_count 不显示
  })
})
