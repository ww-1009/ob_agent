<script setup>
import { onMounted, onUnmounted, ref } from 'vue'
import { fetchHealth } from '../api/health.js'

const health = ref(null)
let timer = null

async function refresh() {
  health.value = await fetchHealth()
}

onMounted(() => {
  refresh()
  timer = setInterval(refresh, 300000)
})
onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="health-badge" :class="{ offline: !health }">
    <template v-if="health">
      {{ health.ocp_provider }}/{{ health.sql_provider }} · {{ health.llm_configured ? 'LLM 就绪' : 'LLM 未配置' }}
    </template>
    <template v-else>后端离线</template>
  </div>
</template>
