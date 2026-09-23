<script setup>
import { computed } from 'vue'
import { formatInt, formatPct, planRows, planStats } from '../lib/plan.js'

const props = defineProps({
  plan: { type: Object, default: null },
})

const rows = computed(() => planRows(props.plan))
const stats = computed(() => planStats(props.plan))
</script>

<template>
  <!-- 只在有归一化计划时渲染；plan 缺字段/为空由 planStats 兜住返回 null -->
  <div v-if="stats" class="plan">
    <div class="plan-head">
      <span class="plan-title">执行计划</span>
      <span class="plan-uid mono">{{ plan.uid || plan.sql_id || '—' }}</span>
      <span class="plan-sub">
        {{ stats.nodeCount }} 个算子 · 累计代价 {{ formatInt(stats.totalCost) }}
      </span>
      <span v-if="stats.truncated" class="plan-warn">节点过多，已截断</span>
    </div>

    <ul class="plan-tree">
      <li
        v-for="r in rows"
        :key="r.id"
        class="plan-node"
        :class="{ hottest: r.hottest }"
        :style="{ paddingLeft: r.indent }"
      >
        <div class="plan-line">
          <span class="plan-op mono">{{ r.operator }}</span>
          <span v-if="r.name" class="plan-name">{{ r.name }}</span>
          <span class="plan-nums">
            行 {{ formatInt(r.rows) }} · 代价 {{ formatInt(r.cost) }}
            <span v-if="r.costPct !== null" class="plan-pct">{{ formatPct(r.costPct) }}</span>
          </span>
        </div>
        <div v-if="r.property" class="plan-prop mono">{{ r.property }}</div>
      </li>
    </ul>

    <table v-if="stats.summary.length" class="plan-summary">
      <thead>
        <tr><th>算子</th><th>次数</th><th>行数</th><th>代价</th><th>占比</th></tr>
      </thead>
      <tbody>
        <tr v-for="s in stats.summary" :key="s.operator">
          <td class="plan-summary-op mono">{{ s.operator }}</td>
          <td>{{ formatInt(s.count) }}</td>
          <td>{{ formatInt(s.rows) }}</td>
          <td>{{ formatInt(s.cost) }}</td>
          <td>{{ formatPct(s.costPct) }}</td>
        </tr>
      </tbody>
    </table>
  </div>
</template>