<template>
  <div id="load-page">
    <projects-wrapper></projects-wrapper>
  </div>
</template>

<script>
import ProjectsWrapper from './projects_wrapper.vue'

export default {
  name: 'LoadPage',
  components: {
    'projects-wrapper': ProjectsWrapper
  }
}
</script>

<style lang="scss">
#load-page {
  width: 100%;
  height: 100%;
}


</style>
