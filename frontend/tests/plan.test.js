import { describe, expect, it } from 'vitest'
import { formatInt, formatPct, planRows, planStats } from '../src/lib/plan.js'

// 与 backend/data/ocp_sql_explain.json 同构（后端 plan_view 归一化后的形态）
const PLAN = {
  uid: 'plan-8f3a1c02',
  sql_id: 'sq-scan-orders-1',
  roots: ['1'],
  nodes: [
    { id: '1', depth: 0, operator: 'EXCHANGE OUT', name: 'distributed', rows: 1280, cost: 48230, property: 'distribution method: HASH' },
    { id: '0', depth: 1, operator: 'TABLE SCAN', name: 'orders', rows: 1280, cost: 48210, property: 'table: tpcc.orders' },
  ],
  summary: [
    { operator: 'TABLE SCAN', count: 1, rows: 1280, cost: 48210 },
    { operator: 'EXCHANGE OUT', count: 1, rows: 1280, cost: 20 },
  ],
  node_count: 2,
  truncated: false,
}

describe('formatInt / formatPct', () => {
  it('千分位；非有限数一律 —', () => {
    expect(formatInt(1280)).toBe('1,280')
    expect(formatInt(48210.9)).toBe('48,211')
    expect(formatInt(null)).toBe('—')
    expect(formatInt(undefined)).toBe('—')
    expect(formatInt(NaN)).toBe('—')
  })
  it('百分比保留一位小数；非有限数为 —', () => {
    expect(formatPct(0.5)).toBe('50.0%')
    expect(formatPct(null)).toBe('—')
  })
})

describe('planStats', () => {
  it('汇总节点数/累计代价/最大行数/最贵算子/算子表占比', () => {
    const s = planStats(PLAN)
    expect(s.nodeCount).toBe(2)
    expect(s.totalCost).toBe(96440)
    expect(s.maxRows).toBe(1280)
    expect(s.hottestId).toBe('1') // 48230 > 48210
    expect(s.truncated).toBe(false)
    expect(s.summary).toHaveLength(2)
    expect(s.summary[0].costPct).toBeCloseTo(48210 / 96440, 6)
  })
  it('缺 node_count 时回退为节点个数；缺 summary 为空表', () => {
    const s = planStats({ nodes: [{ id: 1, operator: 'A' }] })
    expect(s.nodeCount).toBe(1)
    expect(s.summary).toEqual([])
    expect(s.totalCost).toBe(0)
    expect(s.hottestId).toBeNull()
  })
  it('没有归一化计划时返回 null', () => {
    expect(planStats(null)).toBeNull()
    expect(planStats({})).toBeNull()
    expect(planStats({ nodes: [] })).toBeNull()
    expect(planStats({ nodes: 'oops' })).toBeNull()
  })
  it('truncated 只在真为 true 时置位', () => {
    expect(planStats({ ...PLAN, truncated: true }).truncated).toBe(true)
    expect(planStats({ ...PLAN, truncated: 'yes' }).truncated).toBe(false)
  })
})

describe('planRows', () => {
  it('depth → 缩进，代价占比与最贵标记', () => {
    const rows = planRows(PLAN)
    expect(rows.map((r) => [r.operator, r.indent])).toEqual([
      ['EXCHANGE OUT', '0px'],
      ['TABLE SCAN', '16px'],
    ])
    expect(rows[0].hottest).toBe(true)
    expect(rows[1].hottest).toBe(false)
    expect(rows[1].costPct).toBeCloseTo(48210 / 96440, 6)
    expect(rows[1].property).toBe('table: tpcc.orders')
  })
  it('缺字段不渲染成 undefined：rows/cost 为 null、property 为空串', () => {
    const rows = planRows({ nodes: [{ depth: 2, operator: 'TABLE SCAN' }] })
    expect(rows[0]).toMatchObject({
      indent: '32px',
      name: '',
      rows: null,
      cost: null,
      costPct: null,
      property: '',
      hottest: false, // 全无代价时不误标最贵
    })
  })
  it('没有计划时返回空数组', () => {
    expect(planRows(null)).toEqual([])
    expect(planRows({ nodes: [] })).toEqual([])
  })
})