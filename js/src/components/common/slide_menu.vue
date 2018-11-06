<template>
  <div id='slide-menu' v-if='getShowSlideMenu' @click="showSlideMenu(false)">
    <div id='menu-list'>
    </div>
  </div>
</template>

<script>
import {mapGetters, mapMutations} from 'vuex'

export default {
  name: 'SlideMenu',
  computed: {
    ...mapGetters(['getShowSlideMenu']),
  },
  created: function () {
  },
  methods: {
    ...mapMutations(['showSlideMenu']),
  }
}
</script>

<style lang='scss'>
#slide-menu {
  position: fixed;
  left: 0;
  top: 0;
  height: 100vh;
  width: 100%;
  background-color: $slide-window-mask-color;
}

</style>
