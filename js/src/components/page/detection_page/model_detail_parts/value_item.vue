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


.value-item {
  display: flex;
  margin: 4px;
  font-size: $content-inner-box-font-size;
}

.label {
  margin-right: 8px;
  color: #999999;
  font-size: $content-inner-box-font-size;
}

.value {
  font-size: $content-inner-box-font-size;
}
</style>
