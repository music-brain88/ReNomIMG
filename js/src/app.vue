<template>
  <div id="app">
    <app-header/>
    <slide-menu/>
    <modal/>
    <alert-modal v-if="$store.state.alert_modal_flag"></alert-modal>
    <div id="app-content">
      <div id="container">
        <router-view></router-view> 
      </div>
    </div>

  </div>
</template>

<script>
import AlertModal from '@/components/common/alert_modal.vue'
import AppHeader from '@/components/common/app_header.vue'
import SlideMenu from '@/components/common/slide_menu.vue'
import Modal from '@/components/common/modalbox.vue'

export default {
  name: 'App',
  components: {
    'app-header': AppHeader,
    'slide-menu': SlideMenu,
    'modal': Modal,
  },
  created: function () {
    this.$store.commit('setPageName', {
      'page_name': this.$router.currentRoute.name
    })
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
  -webkit-box-sizing: border-box;
  -moz-box-sizing: border-box;
  -o-box-sizing: border-box;
  -ms-box-sizing: border-box;
  box-sizing: border-box;
  font-family: $component-font-family;
}
#app {
  position: relative;
  width: 100%;

  #app-content {
    width: $app-window-width;
    margin: 0 auto;
    padding-top: $app-container-padding-top;
    padding-bottom: $app-container-padding-bottom;
  }
}
</style>
