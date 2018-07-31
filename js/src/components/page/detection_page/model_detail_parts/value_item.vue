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
  name: 'ValueItem',
  props: {
    'label': {
      type: String,
      required: true
    },
    'val': {
      required: true
    }
  }
}
</script>

<style lang="scss" scoped>
$item-font-size: 12px;
$content-margin: 8px;

.value-item {
  display: flex;
  margin: 4px;
  font-size: $item-font-size;
}

.label {
  margin-right: $content-margin;
  color: #999999;
  font-size: $item-font-size;
}

.value {
  font-size: $item-font-size;
}
</style>
