<template>
  <div class="yolo-params">
    <value-item :label="'Horizontal Cell'" :val="params['horizontal_cells']"></value-item>
      <value-item :label="'Vertical Cell'" :val="params['vertical_cells']"></value-item>
      <value-item :label="'Bounding Box'" :val="params['bounding_box']"></value-item>
  </div>
</template>

<script>
import ValueItem from './value_item.vue'

export default {
  name: "YoloParams",
  components: {
    "value-item": ValueItem,
  },
  props: {
    "params": {
      type: Object,
      required: true
    }
  },
}
</script>

<style lang="scss" scoped>
</style>
