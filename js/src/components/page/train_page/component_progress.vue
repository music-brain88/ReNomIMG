<template>
  <component-frame :width-weight="4" :height-weight="3">
    <template slot="header-slot">
      Train Progress
    </template>
    <div id="alg-list">
    </div>
    <div id="component-progress" class="scrollbar-container">
      <progress-bar :isTitle="true"/>
      <progress-bar v-for="item in getRunningModelList" :model="item"/>
    </div>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex'
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
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
  #component-progress {
    height: 100%;
    overflow: auto;
  }
</style>
