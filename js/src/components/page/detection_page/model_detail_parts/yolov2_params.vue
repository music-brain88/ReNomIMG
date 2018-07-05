<template>
  <div class="yolov2-params">
    <value-item :label="'Anchors'" :val="params['anchor']"></value-item>
  </div>
</template>

<script>
import ValueItem from './value_item.vue'

export default {
  name: 'Yolov2Params',
  components: {
    'value-item': ValueItem
  },
  props: {
    'params': {
      type: Object,
      required: true
    }
  }
}
</script>

<style lang="scss" scoped>
</style>
