<template>
  <component-frame :width-weight="8" :height-weight="3">
    <template slot="header-slot">
      Train Progress
    </template>
    <div id="component-progress">
      <progress-bar/>
      <progress-bar/>
      <progress-bar/>
      <progress-bar/>
    </div>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex';
import ComponentFrame from '@/components/common/component_frame.vue'
import ProgressBar from '@/components/page/train_page/progress_bar.vue'

export default {
  name: 'ComponentProgress',
  components: {
    'component-frame': ComponentFrame,
    'progress-bar': ProgressBar
  },
  computed: {
    ...mapGetters(['getRunningModelList'])
  },
  created: function() {

  },
  methods: {

  },
}
</script>

<style lang='scss'>
</style>
