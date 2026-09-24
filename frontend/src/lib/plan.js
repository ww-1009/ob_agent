// 执行计划可视化纯逻辑（与 Vue 解耦，便于单测）。
//
// 后端 app/agent/plan_view.py 已把 OCP 的计划报文归一化成：
//   { uid, sql_id, roots, nodes: [{id, depth, operator, name, rows, cost, property}],
//     summary: [{operator, count, rows, cost}], node_count, truncated }
// nodes 是「先序 + 带 depth」的扁平数组，因此前端只需按 depth 缩进即可还原树形。

const EMPTY = Object.freeze([])

// 非有限数一律显示 —（OCP 常给 null/缺字段，别把 undefined 渲染成 "undefined"）
export function formatInt(v) {
  return Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—'
}

export function formatPct(v) {
  return Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : '—'
}

function nodesOf(plan) {
  if (!plan || typeof plan !== 'object' || !Array.isArray(plan.nodes)) return EMPTY
  return plan.nodes.filter((n) => n && typeof n === 'object')
}

// 头部统计 + 算子汇总表（含每个算子占总代价的比例）
export function planStats(plan) {
  const nodes = nodesOf(plan)
  if (!nodes.length) return null
  let totalCost = 0
  let maxRows = 0
  let hottestId = null
  let hottestCost = -Infinity
  for (const n of nodes) {
    if (Number.isFinite(n.cost)) {
      totalCost += n.cost
      if (n.cost > hottestCost) {
        hottestCost = n.cost
        hottestId = n.id ?? null
      }
    }
    if (Number.isFinite(n.rows)) maxRows = Math.max(maxRows, n.rows)
  }
  const summary = (Array.isArray(plan.summary) ? plan.summary : [])
    .filter((s) => s && typeof s === 'object')
    .map((s) => ({
      operator: String(s.operator ?? '—'),
      count: Number.isFinite(s.count) ? s.count : null,
      rows: Number.isFinite(s.rows) ? s.rows : null,
      cost: Number.isFinite(s.cost) ? s.cost : null,
      costPct: totalCost > 0 && Number.isFinite(s.cost) ? s.cost / totalCost : null,
    }))
  return {
    nodeCount: Number.isFinite(plan.node_count) ? plan.node_count : nodes.length,
    totalCost,
    maxRows,
    hottestId,
    truncated: plan.truncated === true,
    summary,
  }
}

// 树行：depth → indent，代价 → 占全树比例（最贵的算子标 hottest）
export function planRows(plan) {
  const stats = planStats(plan)
  if (!stats) return EMPTY
  return nodesOf(plan).map((n) => {
    const depth = Number.isFinite(n.depth) ? Math.max(0, n.depth) : 0
    const id = String(n.id ?? '')
    return {
      id,
      depth,
      indent: `${depth * 16}px`,
      operator: String(n.operator ?? '—'),
      name: String(n.name ?? ''),
      rows: Number.isFinite(n.rows) ? n.rows : null,
      cost: Number.isFinite(n.cost) ? n.cost : null,
      costPct: stats.totalCost > 0 && Number.isFinite(n.cost) ? n.cost / stats.totalCost : null,
      property: String(n.property ?? ''),
      hottest: stats.hottestId !== null && id === String(stats.hottestId),
    }
  })
}