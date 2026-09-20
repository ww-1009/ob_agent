export async function fetchHealth() {
  try {
    const resp = await fetch('/api/health')
    if (!resp.ok) return null
    return await resp.json()
  } catch {
    return null // 后端不可达 → 离线态
  }
}
