<script setup>
import { toolMeta } from '../composables/chatCore.js'

defineProps({
  events: { type: Array, default: () => [] },
  loading: { type: Boolean, default: false },
  // 'current'：仅当前会话；'all'：全部会话
  scope: { type: String, default: 'current' },
})
const emit = defineEmits(['close', 'refresh', 'toggle-scope'])

function fmtTime(iso) {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const p = (n) => String(n).padStart(2, '0')
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
}

function argsText(args) {
  const a = args || {}
  const keys = Object.keys(a)
  if (!keys.length) return '（无入参）'
  return keys.map((k) => `${k}: ${a[k]}`).join('\n')
}

function stateOf(e) {
  if (e.approved === false) return { text: '已拒绝', cls: 'denied' }
  if (!e.ok) return { text: '失败', cls: 'failed' }
  return { text: '成功', cls: 'ok' }
}
</script>

<template>
  <aside class="audit-panel">
    <div class="audit-head">
      <span class="audit-title">工具审计（{{ events.length }}）</span>
      <span class="audit-actions">
        <button type="button" class="btn-ghost" @click="emit('toggle-scope')">
          {{ scope === 'current' ? '查看全部会话' : '仅看本会话' }}
        </button>
        <button type="button" class="btn-ghost" @click="emit('refresh')">刷新</button>
        <button type="button" class="btn-ghost" @click="emit('close')">关闭</button>
      </span>
    </div>
    <p v-if="loading" class="audit-hint">加载中…</p>
    <p v-else-if="!events.length" class="audit-hint">暂无记录</p>
    <ul v-else class="audit-list">
      <li v-for="e in events" :key="e.id" class="audit-item" :class="stateOf(e).cls">
        <div class="audit-line">
          <span class="audit-state">{{ stateOf(e).text }}</span>
          <span class="audit-tool">{{ e.label || e.tool }}</span>
          <span class="audit-meta">{{ fmtTime(e.created_at) }} · {{ toolMeta(e) }}</span>
        </div>
        <div v-if="e.error" class="audit-error">⚠ {{ e.error }}</div>
        <pre class="audit-args">{{ argsText(e.args) }}</pre>
      </li>
    </ul>
  </aside>
</template>
