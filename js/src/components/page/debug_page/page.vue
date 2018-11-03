<!-->
  This is a page for debugging.
<-->
<template>
  <div id="debug_page">
    <component-progress/>
    <component-scatter/>

    <component-frame :width-weight="4" :height-weight="4">
      This is size(3, 4)
    </component-frame>

    <component-frame :width-weight="2" :height-weight="4">
      This is size(3, 4)
    </component-frame>

    <component-frame :width-weight="10" :height-weight="4">
      This is size(3, 4)
    </component-frame>
    <component-frame :width-weight="2" :height-weight="4">
      This is size(3, 4)
    </component-frame>
    <component-frame :width-weight="2" :height-weight="4">
      This is size(3, 4)
    </component-frame>

    <component-progress/>
    <component-progress/>
  </div>
</template>

<script>

import ComponentFrame from '@/components/common/component_frame.vue'
import ComponentProgress from '@/components/page/train_page/component_progress.vue'
import ComponentScatter from '@/components/page/train_page/component_scatter.vue'

export default {
  name: 'DebugPage',
  components: {
    'component-frame': ComponentFrame,
    'component-progress': ComponentProgress,
    'component-scatter': ComponentScatter,
  },
  created: function() {

  }
}
</script>

<style lang="scss">
#debug_page {
  display: flex;
  align-content: flex-start;
  flex-wrap: wrap;
}
</style>

