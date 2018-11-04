<template>
  <component-frame :width-weight="12" :height-weight="1">
    <template slot="header-slot">
      Tag List
    </template>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentTagList',
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
