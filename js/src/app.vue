<template>
  <div id="app">
    <app-header/>
    <div id="app-content">
      <slide-menu/>
      <div id="container">
        <transition name="fade">
          <router-view></router-view> 
        </transition>
      </div>
    </div>
    <alert-modal v-if="$store.state.show_alert_modal"></alert-modal>
    <modal/>
  </div>
</template>

<script>
import { PAGE_ID } from '@/const.js'
import AlertModal from '@/components/common/alert_modal.vue'
import AppHeader from '@/components/common/app_header.vue'
import SlideMenu from '@/components/common/slide_menu.vue'
import Modal from '@/components/common/modalbox.vue'
import { mapMutations, mapActions } from 'vuex'

export default {
  name: 'App',
  components: {
    'modal': Modal,
    'app-header': AppHeader,
    'slide-menu': SlideMenu,
    'alert-modal': AlertModal
  },
  created: function () {
    this.init()
  },
  methods: {
    ...mapActions(['init']),
    ...mapMutations(['setCurrentPage']),
  }
}
</script>

<style lang="scss">
@font-face{
  font-family: $header-product-name-font-family;
  src: url("../static/fonts/OpenSans-Regular.ttf");
}

@font-face{
  font-family: $component-header-font-family;
  src: url("../static/fonts/OpenSans-Light.ttf");
}

* {
  user-select: none;
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  -o-box-sizing: border-box;
  -ms-box-sizing: border-box;
  box-sizing: border-box;
  font-family: $component-font-family;
}

*:hover {
  transition: all 300ms 0s ease;
}

#app {
  width: 100%;
  #app-content {
    width: $app-window-width;
    margin: $header-height auto;
    padding-top: $app-container-padding-top;
    padding-bottom: $app-container-padding-bottom;
  }
}
</style>
