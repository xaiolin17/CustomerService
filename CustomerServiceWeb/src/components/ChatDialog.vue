<template>
  <div v-if="visible" class="chat-dialog-overlay" @click.self="$emit('close')">
    <div class="chat-dialog">
      <!-- 标题栏 -->
      <div class="chat-header">
        <span class="chat-title">智能客服</span>
        <button class="close-btn" @click="$emit('close')" aria-label="关闭">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      <!-- 消息区域 -->
      <div class="chat-messages" ref="messagesRef">
        <div
          v-for="(msg, index) in messages"
          :key="index"
          :class="['message', msg.role === 'user' ? 'message-user' : 'message-assistant']"
        >
          <div class="message-bubble">{{ msg.content }}</div>
        </div>
      </div>

      <!-- 输入区域 -->
      <div class="chat-input-area">
        <input
          v-model="inputText"
          type="text"
          class="chat-input"
          placeholder="请输入您的问题..."
          @keydown.enter="sendMessage"
        />
        <button class="send-btn" @click="sendMessage" :disabled="!inputText.trim()">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          >
            <line x1="22" y1="2" x2="11" y2="13" />
            <polygon points="22 2 15 22 11 13 2 9 22 2" />
          </svg>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  visible: {
    type: Boolean,
    default: false
  },
  messages: {
    type: Array,
    default: () => []
  }
})

const emit = defineEmits(['close', 'send'])

const inputText = ref('')
const messagesRef = ref(null)

function sendMessage() {
  if (!inputText.value.trim()) return
  emit('send', inputText.value)
  inputText.value = ''
}

watch(
  () => props.messages.length,
  async () => {
    await nextTick()
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  }
)
</script>

<style scoped>
.chat-dialog-overlay {
  position: fixed;
  inset: 0;
  background-color: rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1001;
}

.chat-dialog {
  width: 400px;
  height: 550px;
  background-color: #fff;
  border-radius: 12px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.chat-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 20px;
  border-bottom: 1px solid #f0f0f0;
  background-color: #fff;
}

.chat-title {
  font-size: 16px;
  font-weight: 600;
  color: #333;
}

.close-btn {
  width: 32px;
  height: 32px;
  border: none;
  background: none;
  cursor: pointer;
  border-radius: 4px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #999;
  transition: all 0.2s;
}

.close-btn:hover {
  background-color: #f5f5f5;
  color: #333;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  background-color: #fafafa;
}

.message {
  display: flex;
}

.message-user {
  justify-content: flex-end;
}

.message-assistant {
  justify-content: flex-start;
}

.message-bubble {
  max-width: 80%;
  padding: 10px 14px;
  border-radius: 8px;
  font-size: 14px;
  line-height: 1.5;
  word-break: break-word;
}

.message-user .message-bubble {
  background-color: #1677ff;
  color: #fff;
  border-bottom-right-radius: 2px;
}

.message-assistant .message-bubble {
  background-color: #f0f0f0;
  color: #333;
  border-bottom-left-radius: 2px;
}

.chat-input-area {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid #f0f0f0;
  background-color: #fff;
}

.chat-input {
  flex: 1;
  height: 40px;
  border: 1px solid #d9d9d9;
  border-radius: 6px;
  padding: 0 12px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.chat-input:focus {
  border-color: #1677ff;
}

.chat-input::placeholder {
  color: #bfbfbf;
}

.send-btn {
  width: 40px;
  height: 40px;
  border: none;
  background-color: #1677ff;
  color: #fff;
  border-radius: 6px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.2s;
  flex-shrink: 0;
}

.send-btn:hover:not(:disabled) {
  background-color: #4096ff;
}

.send-btn:disabled {
  background-color: #d9d9d9;
  cursor: not-allowed;
}
</style>