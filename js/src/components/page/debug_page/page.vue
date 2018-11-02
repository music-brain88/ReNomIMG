<!-->
  This is a page for debugging.
<-->
<template>
  <div id="debug_page">
    <component-frame :width-weight="2" :height-weight="3">
      This is size(2, 3)
    </component-frame>
    <component-frame :width-weight="3" :height-weight="4">
      This is size(3, 4)
    </component-frame>
    <component-frame :width-weight="6" :height-weight="4">
      This is size(3, 4)
    </component-frame>
  </div>
</template>

<script>

import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'DebugPage',
  components: {
    'component-frame': ComponentFrame
  },
  created: {

  }
}
</script>

<style lang="scss">
#debug_page {
  display: flex;
  align-content: flex-start;
}
</style>

