/** A random id made on this device. The server only ever stores a salted hash of it. */
export function deviceId(): string {
  const key = 'curbcut.device'
  try {
    const saved = localStorage.getItem(key)
    if (saved) return saved
    const fresh = crypto.randomUUID()
    localStorage.setItem(key, fresh)
    return fresh
  } catch {
    return 'session-' + Math.random().toString(36).slice(2, 14).padEnd(12, '0')
  }
}
