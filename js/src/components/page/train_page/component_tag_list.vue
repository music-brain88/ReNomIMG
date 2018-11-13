<template>
  <component-frame :width-weight="6" :height-weight="2.6">
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
    ...mapGetters(['getTagList']),
  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
</style>
