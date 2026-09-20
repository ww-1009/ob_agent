<script setup>
defineEmits(['pick'])

const samples = [
  '有哪些慢SQL？帮我分析一下',
  '查看集群拓扑与租户',
  '帮我 explain 一条慢SQL并给优化建议',
]
</script>

<template>
  <div class="chips">
    <button v-for="(s, i) in samples" :key="i" type="button" class="chip" @click="$emit('pick', s)">
      {{ s }}
    </button>
  </div>
</template>
