<template>
  <div class="app-container">
    <ChatIcon
      :visible="!isOpen"
      @open="handleOpen"
    />
    <ChatDialog
      :visible="isOpen"
      :messages="messages"
      @close="handleClose"
      @send="handleSend"
    />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import ChatIcon from './components/ChatIcon.vue'
import ChatDialog from './components/ChatDialog.vue'
import { useWebSocket } from './composables/useWebSocket'

const isOpen = ref(false)
const sessionId = ref('')
const messages = ref([])

const ws = useWebSocket(sessionId)

function generateSessionId() {
  const chars = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'
  return chars.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0
    const v = c === 'x' ? r : (r & 0x3) | 0x8
    return v.toString(16)
  })
}

onMounted(() => {
  sessionId.value = generateSessionId()
})

function handleOpen() {
  isOpen.value = true
  ws.connect((msg) => {
    messages.value.push({ role: 'assistant', content: msg })
  })
}

function handleClose() {
  isOpen.value = false
}

function handleSend(text) {
  if (!text.trim()) return
  messages.value.push({ role: 'user', content: text })
  ws.send(text)
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background-color: #f5f5f5;
}

.app-container {
  min-height: 100vh;
  position: relative;
}
</style>