import MarkdownIt from 'markdown-it'
// 用 lib/common 子集（36 语言，含 sql/js/bash/json/xml/python 等）避免打进全套 193 语言
import hljs from 'highlight.js/lib/common'
import createDOMPurify from 'dompurify'

// DOMPurify v3：默认导出是工厂，浏览器/jsdom(window 存在)下传入 window 即得实例。
const DOMPurify = createDOMPurify(window)

function esc(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

// 自定义 fence：highlight.js 高亮 + 代码块头部（语言名 + 复制按钮，源码放 data-code）
function fenceRenderer(tokens, idx) {
  const info = tokens[idx].info ? tokens[idx].info.trim() : ''
  const lang = (info.split(/\s+/)[0] || '').toLowerCase()
  const raw = tokens[idx].content
  let code = ''
  let cls = ''
  if (lang && hljs.getLanguage(lang)) {
    try {
      code = hljs.highlight(raw, { language: lang, ignoreIllegals: true }).value
      cls = ' language-' + lang
    } catch { code = esc(raw) }
  } else {
    try { code = hljs.highlightAuto(raw).value } catch { code = esc(raw) }
  }
  const label = lang || 'code'
  return (
    '<div class="codeblock">' +
    '<div class="codeblock-head"><span class="codeblock-lang">' + esc(label) + '</span>' +
    '<button type="button" class="copy-btn" data-code="' + esc(raw) + '">复制</button></div>' +
    '<pre class="hljs' + cls + '"><code>' + code + '</code></pre>' +
    '</div>'
  )
}

const md = new MarkdownIt({ html: false, linkify: true, breaks: false })
md.renderer.rules.fence = fenceRenderer

export function renderMarkdown(text) {
  return DOMPurify.sanitize(md.render(text || ''))
}
