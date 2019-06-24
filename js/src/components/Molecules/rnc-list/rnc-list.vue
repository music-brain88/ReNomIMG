<template>
  <div
    id="tag-list"
    class="scrollbar-container"
  >
    <component
      v-for="(item, index) in ItemArray"
      :is="tagKind"
      :key="index"

      :tag-name="item"
      :tag-id="index"

      :model="item.Model"
      :last-batch-loss="item.LastBatchLoss"
      :result-of-metric1="item.ResultOfMetric1"
      :result-of-metric2="item.ResultOfMetric2"
      :selected-model-id="item.SelectedModelId"

      @clicked-model-id="$emit('clicked-model-id', $event)"
      @rm-model="$emit('rm-model', $event)"
    />
  </div>
</template>

<script>
import RncTagListItem from './../../Atoms/rnc-tag-list-item/rnc-tag-list-item.vue'
import RncModelListItem from './../../Molecules/rnc-model-list-item/rnc-model-list-item.vue'

export default {
  name: 'RncList',
  components: {
    'rnc-tag-list-item': RncTagListItem,
    'rnc-model-list-item': RncModelListItem
  },
  props: {
    ItemArray: {
      type: Array,
      default: function () { return [] }
    },
    kind: {
      type: String,
      default: 'tag',
      // ▼ Write the component to rotate in the list
      validator: val => ['tag', 'model'].includes(val)
    }
  },
  computed: {
    tagKind () {
      return `rnc-${this.kind}-list-item`
    }
  }
}
</script>

<style lang="scss" scoped>
@import './../../../../static/css/unified.scss';

#tag-list {
  width: 100%;
  height: 100%;
  padding-right: $padding-micro;
}
</style>
