<template>
  <component-frame :width-weight="12" :height-weight="3.5">
    <template slot="header-slot">
      Prediction Result
    </template>
    AA
  </component-frame>
</template>

<script>
import {mapGetters, mapState} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentPredictionResult',
  components: {
    'component-frame': ComponentFrame
  },
  computed: {
    ...mapState(['datasets', 'getSelectedModel']),
    getValidImages: function () {
      const model = this.getSelectedModel
      if (model) {
        const dataset = this.datasets.filter(d => d.id === model.dataset_id)
        const valid_data = dataset.valid_data
        return ['a']
      }
      return []
    }
  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
</style>
