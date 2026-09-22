<script setup>
import { nextTick, onMounted, ref, watch } from 'vue'
import HealthBadge from './components/HealthBadge.vue'
import QuickChips from './components/QuickChips.vue'
import MessageBubble from './components/MessageBubble.vue'
import ConfirmDialog from './components/ConfirmDialog.vue'
import ThreadSidebar from './components/ThreadSidebar.vue'
import { useChat } from './composables/useChat.js'

const {
  messages, busy, canSend, llmNotConfigured, pendingConfirm, send, stop, clear, decideConfirm,
  memoryEnabled, threads, loadingThread, threadId, init, openThread, newThread, removeThread,
} = useChat()
const draft = ref('')
const scrollEl = ref(null)
const atBottom = ref(true)
const MAX_INPUT = 2000

// 启动即对齐服务端能力：记忆开启时恢复上次会话与历史列表
onMounted(() => {
  void init()
})

async function submit() {
  if (busy.value) return // busy 时（回答中）不允许 Enter 新发送；按钮此时已是「停止」
  const text = draft.value.trim()
  if (!text) return
  if (text.length > MAX_INPUT) {
    alert(`输入过长（最多 ${MAX_INPUT} 字符）`)
    return
  }
  atBottom.value = true // 发送 = 恢复跟随底部（deep watch 负责实际滚动）
  draft.value = ''
  await send(text)
}

function onPick(text) {
  draft.value = text
  void submit()
}

async function onSelectThread(id) {
  if (busy.value) stop() // 切换会话即中止在途回答（后端会把已收到的部分落库）
  atBottom.value = true
  await openThread(id)
}

function onScroll() {
  const el = scrollEl.value
  if (!el) return
  atBottom.value = el.scrollHeight - el.scrollTop - el.clientHeight < 80
}

async function scrollToBottom() {
  await nextTick()
  const el = scrollEl.value
  if (el) {
    el.scrollTop = el.scrollHeight
    atBottom.value = true
  }
}

function onEnter(e) {
  // IME 组合中（含部分浏览器 keyCode 229）：候选确认回车交给输入法，不发送
  if (e.isComposing || e.keyCode === 229) return
  e.preventDefault()
  submit()
}

function autosize(e) {
  const el = e.target
  el.style.height = 'auto'
  el.style.height = Math.min(el.scrollHeight, 180) + 'px'
}

watch(
  () => messages.value,
  async (msgs) => {
    if (msgs.length && atBottom.value) await scrollToBottom()
  },
  { deep: true },
)
</script>

<template>
  <div class="app-shell">
    <ThreadSidebar
      v-if="memoryEnabled"
      :threads="threads"
      :active-id="threadId"
      :loading="loadingThread"
      @select="onSelectThread"
      @create="newThread"
      @remove="removeThread"
    />

    <div class="app">
      <header class="topbar">
        <h1 class="title">OceanBase DB Agent</h1>
        <div class="topbar-right">
          <!-- 侧栏已提供「新对话」入口，无侧栏（未启用记忆）时保留顶栏按钮 -->
          <button v-if="messages.length && !memoryEnabled" type="button" class="btn-ghost" @click="clear">新对话</button>
          <HealthBadge />
        </div>
      </header>

      <div v-if="llmNotConfigured" class="notice warn">
        LLM 未配置：请在 <code>backend/config.yaml</code> 填写 <code>llm.base_url / api_key / model</code> 后重启后端。
      </div>

      <main ref="scrollEl" class="thread" @scroll="onScroll">
        <section v-if="loadingThread && !messages.length" class="empty">
          <p class="empty-tip">正在加载历史会话…</p>
        </section>
        <section v-else-if="!messages.length" class="empty">
          <p class="empty-hello">你好，我是 OceanBase DBA 助手。</p>
          <p class="empty-tip">
            排查慢SQL、分析执行计划、优化查询性能。交互内容会发送到所配置的 LLM API。
          </p>
          <QuickChips @pick="onPick" />
        </section>
        <section v-else class="list">
          <MessageBubble v-for="m in messages" :key="m.id" :message="m" />
        </section>
      </main>

      <footer class="composer">
        <textarea
          v-model="draft"
          class="input"
          rows="1"
          :maxlength="MAX_INPUT"
          placeholder="输入问题，Enter 发送，Shift+Enter 换行"
          @input="autosize"
          @keydown.enter.exact="onEnter"
        ></textarea>
        <div class="composer-actions">
          <span class="counter">{{ draft.length }}/{{ MAX_INPUT }}</span>
          <button v-if="busy" type="button" class="btn primary" @click="stop">停止</button>
          <button v-else type="button" class="btn primary" :disabled="!canSend || !draft.trim()" @click="submit">发送</button>
        </div>
      </footer>

      <ConfirmDialog :confirm="pendingConfirm" @decide="decideConfirm" />
    </div>
  </div>
</template>
