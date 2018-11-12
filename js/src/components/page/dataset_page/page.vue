<template>
  <div id="dataset-page">
    <div id="components">
      <component-dataset/>
    </div>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentDataset from '@/components/page/dataset_page/component_dataset.vue'

export default {
  name: 'DatasetPage',
  components: {
    'component-dataset': ComponentDataset,
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
#dataset-page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
  #components {
    display: flex;
    align-content: flex-start;
    flex-wrap: wrap;
    width: calc(12*#{$component-block-width});
  }  
}
</style>
