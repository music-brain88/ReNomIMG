<template>
  <component-frame
    :width-weight="12"
    :height-weight="9">
    <template slot="header-slot">
      Dataset
    </template>
    <div id="dataset">
      <div id="dataset-list">
        <table-item :is-title="true"/>
        <table-item
          v-for="(item, key) in getFilteredDatasetList"
          :key="key"
          :dataset="item"/>
      </div>
      <div id="dataset-content"/>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters } from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import DatasetTableItem from '@/components/page/dataset_page/table_item.vue'

export default {
  name: 'ComponentDataset',
  components: {
    'component-frame': ComponentFrame,
    'table-item': DatasetTableItem,
  },
  computed: {
    ...mapGetters(['getFilteredDatasetList']),
  },
  created: function () {

  },
  methods: {
  }
}
</script>

<style lang='scss'>
#dataset {
  display: flex;
  height: 100%;
  width: 100%;
  #dataset-list {
    height: 100%;
    width: 60%;
    padding: $dataset-table-padding;
  }
  #dataset-content {
    height: 100%;
    width: 40%;
  }
}
</style>
