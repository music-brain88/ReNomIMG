<template>
  <div class="value-item">
    <div class="label">
      {{ label }}
    </div>
    <div class="value">
      {{ val }}
    </div>
  </div>
</template>

<script>
export default {
  name: "ValueItem",
  props: {
    "label": {
      type: String,
      required: true
    },
    "val": {
      required: true
    }
  },
}
</script>

<style lang="scss" scoped>
$content-margin: 8px;

.value-item {
  display: flex;
  margin: 4px;
}

.label {
  margin-right: $content-margin;
  color: #999999;
}
</style>
