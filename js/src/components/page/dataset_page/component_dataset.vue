<template>
  <component-frame :width-weight="10" :height-weight="8">
    <template slot="header-slot">
      Dataset
    </template>
    <div id="dataset">

    </div>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentDataset',
  components: {
    'component-frame': ComponentFrame
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
</style>
