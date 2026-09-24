<script setup>
import { ref } from 'vue'
import { toolMeta } from '../composables/chatCore.js'
import PlanView from './PlanView.vue'

defineProps({
  tools: { type: Array, default: () => [] },
})

const openIds = ref(new Set())

function toggle(id) {
  const next = new Set(openIds.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  openIds.value = next
}

function isOpen(id) {
  return openIds.value.has(id)
}

// 已拒绝 与 执行失败 是两种不同的失败：前者是人工否决，后者是工具本身报错
function stateOf(t) {
  if (t.approved === false) return { text: '已拒绝', cls: 'denied' }
  if (!t.ok) return { text: '失败', cls: 'failed' }
  return { text: '成功', cls: 'ok' }
}

function argsText(t) {
  const args = t.args || {}
  const keys = Object.keys(args)
  if (!keys.length) return '（无入参）'
  return keys.map((k) => `${k}: ${args[k]}`).join('\n')
}
</script>

<template>
  <div v-if="tools.length" class="trace">
    <div class="trace-title">工具调用（{{ tools.length }}）</div>
    <div v-for="t in tools" :key="t.id || t.name" class="trace-row" :class="stateOf(t).cls">
      <button type="button" class="trace-head" @click="toggle(t.id || t.name)">
        <span class="trace-caret">{{ isOpen(t.id || t.name) ? '▾' : '▸' }}</span>
        <span class="trace-name">{{ t.label || t.name }}</span>
        <span class="trace-meta">{{ toolMeta(t) }}</span>
        <span class="trace-state">{{ stateOf(t).text }}</span>
      </button>
      <div v-if="isOpen(t.id || t.name)" class="trace-body">
        <div v-if="t.error" class="trace-error">⚠ {{ t.error }}</div>
        <!-- 只有 get_sql_explain 的轨迹带 plan：展示算子树与代价占比 -->
        <PlanView v-if="t.plan" :plan="t.plan" />
        <pre class="trace-args">{{ argsText(t) }}</pre>
      </div>
    </div>
  </div>
</template>
