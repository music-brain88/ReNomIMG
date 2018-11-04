<template>
  <component-frame :width-weight="2" :height-weight="1.5">
    <template slot="header-slot">
      Model Filter
    </template>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentModelFilter',
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
