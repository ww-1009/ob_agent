<script setup>
import { computed } from 'vue'
import { renderMarkdown } from '../lib/markdown.js'

const props = defineProps({
  text: { type: String, default: '' },
})

const html = computed(() => renderMarkdown(props.text))

async function onContainerClick(e) {
  const btn = e.target.closest('.copy-btn')
  if (!btn || btn.disabled) return
  // 同步禁用，堵住 clipboard 权限弹窗期间二次点击进入两个重叠 flash 定时器的窗口
  btn.disabled = true
  // data-code 可能被 DOMPurify mXSS 防御剥除（代码块含完整 <script>…<\/script> 时）
  const fallback = btn.closest('.codeblock')?.querySelector('code')?.textContent ?? ''
  const code = btn.dataset.code ?? fallback
  if (!code) {
    flash(btn, '无可复制内容')
    return
  }
  try {
    await navigator.clipboard.writeText(code)
    flash(btn, '已复制')
  } catch {
    flash(btn, '复制失败')
  }
}

function flash(btn, msg) {
  const old = btn.textContent
  btn.textContent = msg
  btn.disabled = true
  setTimeout(() => {
    btn.textContent = old
    btn.disabled = false
  }, 1200)
}
</script>

<template>
  <div class="md" @click="onContainerClick" v-html="html"></div>
</template>
