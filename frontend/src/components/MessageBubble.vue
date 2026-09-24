<script setup>
import MarkdownBody from './MarkdownBody.vue'
import ToolTrace from './ToolTrace.vue'

defineProps({
  message: { type: Object, required: true },
})
</script>

<template>
  <div class="bubble-row" :class="message.role">
    <div class="bubble" :class="message.role">
      <template v-if="message.role === 'user'">
        <div class="user-text">{{ message.content }}</div>
      </template>
      <template v-else>
        <div v-if="message.error" class="error-banner">⚠ {{ message.error }}</div>
        <div v-if="Array.isArray(message.status) && message.status.length > 0" class="status-line">
          <span v-for="(s, i) in message.status" :key="i">{{ s }}</span>
        </div>
        <MarkdownBody v-if="message.content" :text="message.content" />
        <ToolTrace :tools="message.tools || []" />
        <span v-if="message.state === 'streaming'" class="caret"></span>
      </template>
    </div>
  </div>
</template>
