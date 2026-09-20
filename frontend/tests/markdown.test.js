import { describe, expect, it } from 'vitest'
import { renderMarkdown } from '../src/lib/markdown.js'

describe('renderMarkdown', () => {
  it('渲染标题', () => {
    expect(renderMarkdown('# 标题')).toContain('<h1>标题</h1>')
  })
  it('渲染 GFM 表格', () => {
    const md = '| a | b |\n| - | - |\n| 1 | 2 |'
    const html = renderMarkdown(md)
    expect(html).toContain('<table>')
    expect(html).toContain('<td>1</td>')
  })
  it('html:false 下原始 <script> 被转义而非执行', () => {
    const html = renderMarkdown('<script>alert(1)</script>')
    expect(html).not.toContain('<script>')
    expect(html).not.toContain('onerror')
  })
  it('原始 HTML/img onerror 不生成可执行元素', () => {
    const html = renderMarkdown('<img src=x onerror="alert(1)">')
    const doc = new DOMParser().parseFromString(html, 'text/html')
    expect(doc.querySelector('[onerror]')).toBeNull()
  })
  it('javascript: 不生成可点击链接', () => {
    const html = renderMarkdown('[x](javascript:alert(1))')
    const doc = new DOMParser().parseFromString(html, 'text/html')
    expect(doc.querySelector('a[href^="javascript:"]')).toBeNull()
  })
  it('代码块被高亮并带复制按钮', () => {
    const html = renderMarkdown('```js\nconst a = 1\n```')
    expect(html).toContain('language-js')
    expect(html).toContain('class="copy-btn"')
    expect(html).toContain('hljs')
  })
})
