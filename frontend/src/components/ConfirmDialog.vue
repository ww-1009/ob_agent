<script setup>
import { computed, onUnmounted, ref, watch } from 'vue'

const props = defineProps({
  confirm: { type: Object, default: null },
})
const emit = defineEmits(['decide'])

// 倒计时（秒），基于后端给的 expires_in 与接收时刻推算
const countdown = ref(0)
let timer = 0

function startCountdown(expiresIn, requestedAt) {
  stopCountdown()
  const elapsed = Math.max(0, (Date.now() - (requestedAt || Date.now())) / 1000)
  countdown.value = Math.max(0, expiresIn - elapsed)
  timer = window.setInterval(() => {
    countdown.value = Math.max(0, countdown.value - 1)
    if (countdown.value <= 0) {
      stopCountdown()
      // 后端会超时拒绝并 emit confirm_timeout；本地同步关闭卡片
      emit('decide', false)
    }
  }, 1000)
}

function stopCountdown() {
  if (timer) {
    clearInterval(timer)
    timer = 0
  }
}

watch(
  () => props.confirm,
  (c) => {
    if (c && c.state !== 'deciding') {
      startCountdown(c.expires_in, c.requested_at)
    } else {
      stopCountdown()
    }
  },
  { immediate: true },
)

onUnmounted(stopCountdown)

const left = computed(() => Math.ceil(countdown.value))
const sql = computed(() => String(props.confirm?.args?.sql ?? '').trim())
const ctx = computed(() => {
  const a = props.confirm?.args || {}
  const parts = []
  for (const key of ['tenant_name', 'cluster_name', 'db_name', 'tenant_type']) {
    if (a[key]) parts.push(`${key}=${a[key]}`)
  }
  return parts.join('  ')
})
</script>

<template>
  <div v-if="confirm" class="confirm-overlay">
    <div class="confirm-card">
      <div class="confirm-head">
        <span class="confirm-title">🔒 需要人工审批</span>
        <span class="confirm-tool">{{ confirm.tool_label || confirm.tool }}</span>
      </div>

      <div v-if="ctx" class="confirm-ctx mono">{{ ctx }}</div>
      <pre v-if="sql" class="confirm-sql">{{ sql }}</pre>

      <div class="confirm-foot">
        <span class="confirm-expires">超时 {{ left }}s</span>
        <div class="confirm-actions">
          <button type="button" class="btn danger" :disabled="confirm.state === 'deciding'" @click="emit('decide', false)">
            拒绝
          </button>
          <button type="button" class="btn primary" :disabled="confirm.state === 'deciding'" @click="emit('decide', true)">
            允许
          </button>
        </div>
      </div>
      <div v-if="confirm.error" class="confirm-error">⚠ {{ confirm.error }}</div>
    </div>
  </div>
</template>

<style scoped>
.confirm-overlay {
  position: fixed; inset: 0; z-index: 50;
  display: flex; align-items: center; justify-content: center;
  background: rgba(0, 0, 0, 0.35);
}
.confirm-card {
  width: min(560px, calc(100% - 32px));
  background: var(--panel); border: 1px solid var(--line);
  border-radius: var(--radius); box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
  padding: 16px 18px;
}
.confirm-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: 10px; margin-bottom: 10px;
}
.confirm-title { font-size: 15px; font-weight: 600; }
.confirm-tool { font-size: 13px; padding: 2px 10px; border-radius: 999px; background: var(--panel-2); color: var(--text-2); }
.confirm-ctx { font-size: 12px; color: var(--text-2); margin-bottom: 8px; word-break: break-all; }
.confirm-sql {
  margin: 0 0 12px; padding: 10px; max-height: 220px; overflow: auto;
  background: var(--code-bg); border: 1px solid var(--code-line); border-radius: 8px;
  font-size: 13px; line-height: 1.5; white-space: pre-wrap; word-break: break-all;
}
.confirm-foot {
  display: flex; align-items: center; justify-content: space-between; gap: 12px;
}
.confirm-expires { font-size: 12px; color: var(--text-2); }
.confirm-actions { display: flex; gap: 8px; }
.btn.danger { background: var(--err); }
.confirm-error { margin-top: 10px; font-size: 13px; color: var(--err); }
</style>
