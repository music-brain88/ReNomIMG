<template>
  <div id="table-item">
    <div id="dataset-id">
      <span v-if="isTitle">Model ID</span>
      <span v-else>{{ dataset.id }}</span>
    </div>
    <div id="dataset-name">
      <span v-if="isTitle">Name</span>
      <span v-else>{{ dataset.name }}</span>
    </div>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'DatasetTableItem',
  components: {

  },
  props: {
    dataset: Object,
    isTitle: {
      type: Boolean,
      default: false
    }
  },
  computed: {
  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
#table-item {
  display: flex;
  width: 100%;
  height: $dataset-table-item-height;
  text-align: center;
  #dataset-id {
    width: $dataset-table-id-width;
    height: 100%;
  }
  #dataset-name {
    width: $dataset-table-name-width;
    height: 100%;
  }
}
</style>
