import { describe, expect, it } from 'vitest'
import { buildHistory, dedupeStatus } from '../src/composables/chatCore.js'

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
