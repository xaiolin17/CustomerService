<template>
  <button
    v-if="visible"
    class="chat-icon"
    @click="$emit('open')"
    aria-label="打开智能客服"
  >
    <svg
      width="28"
      height="28"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="2"
      stroke-linecap="round"
      stroke-linejoin="round"
    >
      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
    </svg>
  </button>
</template>

<script setup>
defineProps({
  visible: {
    type: Boolean,
    default: true
  }
})

defineEmits(['open'])
</script>

<style scoped>
.chat-icon {
  position: fixed;
  bottom: 24px;
  right: 24px;
  width: 56px;
  height: 56px;
  border-radius: 50%;
  background-color: #1677ff;
  color: #fff;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(22, 119, 255, 0.4);
  transition: all 0.3s ease;
  z-index: 1000;
}

.chat-icon:hover {
  background-color: #4096ff;
  box-shadow: 0 6px 16px rgba(22, 119, 255, 0.5);
  transform: scale(1.05);
}

.chat-icon:active {
  transform: scale(0.95);
}
</style>