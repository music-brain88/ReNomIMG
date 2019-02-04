<template>
  <div id="table-item">
    <div id="dataset-id">
      <span v-if="isTitle">Data ID</span>
      <span v-else>{{ dataset.id }}</span>
    </div>
    <div id="dataset-name">
      <span v-if="isTitle">Name</span>
      <span v-else>{{ dataset.name }}</span>
    </div>
    <div id="test-dataset-id">
      <span v-if="isTitle">Test</span>
      <span v-else>{{ this.checkTestDataset(dataset.test_dataset_id) }}</span>
    </div>
    <div id="valid-data">
      <span v-if="isTitle">Valid</span>
      <span v-else>{{ dataset.valid_data.target.length }}</span>
    </div>
    <div id="dataset-name">
      <span v-if="isTitle">Created</span>
      <span v-else>{{ Object.keys(dataset) }}</span>
      {{ dataset.class_map }}
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'
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
    checkTestDataset: function (test) {
      return test !== -1 ? test : 'None'
    }
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
  #test-dataset-id {
    width: $dataset-table-test-dataset-id-width;
    height: 100%;
  }
  #valid-data {
    width: $dataset-table-valid-data-width;
    height: 100%;
  }
}
</style>
