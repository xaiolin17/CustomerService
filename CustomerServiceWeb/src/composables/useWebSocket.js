import { ref } from 'vue'

/**
 * 获取后端 WebSocket 基础地址。
 * 优先使用 Vite 环境变量，否则回退到相对路径（通过 Vite proxy 转发）。
 */
function getWsBaseUrl() {
  const envUrl = import.meta.env.VITE_WS_URL
  if (envUrl) {
    return envUrl
  }
  // 通过 Vite proxy 转发（开发环境）
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}`
}

export function useWebSocket(sessionId) {
  let ws = null
  let reconnectTimer = null
  let reconnectAttempts = 0
  const maxReconnectDelay = 30000
  const isConnected = ref(false)

  function getReconnectDelay() {
    return Math.min(1000 * Math.pow(2, reconnectAttempts), maxReconnectDelay)
  }

  function scheduleReconnect(onMessage) {
    if (reconnectTimer) return
    const delay = getReconnectDelay()
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      reconnectAttempts++
      connect(onMessage)
    }, delay)
  }

  function clearReconnect() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
  }

  function connect(onMessage) {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return
    }

    const baseUrl = getWsBaseUrl()
    const url = `${baseUrl}/ws?session_id=${sessionId.value}`
    ws = new WebSocket(url)

    ws.onopen = () => {
      isConnected.value = true
      reconnectAttempts = 0
      clearReconnect()
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.type === 'welcome' && onMessage) {
          onMessage(data.content)
        } else if (data.type === 'reply' && onMessage) {
          onMessage(data.content)
        } else if (data.type === 'info' && onMessage) {
          onMessage(data.content)
        }
      } catch (e) {
        // 非 JSON 格式，直接传递
        if (onMessage) onMessage(event.data)
      }
    }

    ws.onclose = () => {
      isConnected.value = false
      ws = null
      scheduleReconnect(onMessage)
    }

    ws.onerror = () => {
      if (ws) {
        ws.close()
      }
    }
  }

  function send(message) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'chat',
        content: message,
        session_id: sessionId.value
      }))
    }
  }

  function sendClear() {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        type: 'clear',
        session_id: sessionId.value
      }))
    }
  }

  function close() {
    clearReconnect()
    if (ws) {
      ws.onclose = null
      ws.close()
      ws = null
    }
    isConnected.value = false
    reconnectAttempts = 0
  }

  return {
    connect,
    send,
    sendClear,
    close,
    isConnected
  }
}