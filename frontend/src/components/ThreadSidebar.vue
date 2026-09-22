<script setup>
import { threadLabel } from '../composables/chatCore.js'

defineProps({
  threads: { type: Array, default: () => [] },
  activeId: { type: String, default: '' },
  loading: { type: Boolean, default: false },
})
const emit = defineEmits(['select', 'create', 'remove'])

// 相对时间：列表只用于区分新旧，不必精确
function fmtTime(iso) {
  const t = Date.parse(iso)
  if (!Number.isFinite(t)) return ''
  const diff = Date.now() - t
  if (diff < 60_000) return '刚刚'
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)} 分钟前`
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)} 小时前`
  const d = new Date(t)
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}
</script>

<template>
  <aside class="sidebar">
    <div class="sidebar-head">
      <span class="sidebar-title">历史会话</span>
      <button type="button" class="btn-ghost" @click="emit('create')">新对话</button>
    </div>

    <p v-if="loading && !threads.length" class="sidebar-hint">加载中…</p>
    <p v-else-if="!threads.length" class="sidebar-hint">暂无历史会话</p>
    <ul v-else class="sidebar-list">
      <li
        v-for="t in threads"
        :key="t.thread_id"
        class="sidebar-item"
        :class="{ active: t.thread_id === activeId }"
      >
        <button type="button" class="sidebar-open" :title="threadLabel(t)" @click="emit('select', t.thread_id)">
          <span class="sidebar-item-title">{{ threadLabel(t) }}</span>
          <span class="sidebar-item-meta">{{ fmtTime(t.updated_at) }} · {{ t.message_count }} 条</span>
        </button>
        <button type="button" class="sidebar-delete" title="删除该会话" @click="emit('remove', t.thread_id)">×</button>
      </li>
    </ul>
  </aside>
</template>
