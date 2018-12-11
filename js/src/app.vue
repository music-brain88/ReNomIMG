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
    <transition name="fade">
      <div id="loading-whole-mask" v-if="show_loading_mask">
        <div id="loader-container">
          <div class="loader"></div>
          <div class="loader"></div>
          <div class="loader"></div>
          <div class="loader"></div>
          <div class="loader"></div>
          <div class="message">Synchronizing to Server...</div>
        </div>
      </div>
    </transition>
    <app-footer/>
  </div>
</template>

<script>
import { PAGE_ID } from '@/const.js'
import AlertModal from '@/components/common/alert_modal.vue'
import AppHeader from '@/components/common/app_header.vue'
import AppFooter from '@/components/common/app_footer.vue'
import SlideMenu from '@/components/common/slide_menu.vue'
import Modal from '@/components/common/modalbox.vue'
import { mapMutations, mapActions, mapState } from 'vuex'

export default {
  name: 'App',
  components: {
    'modal': Modal,
    'app-header': AppHeader,
    'app-footer': AppFooter,
    'slide-menu': SlideMenu,
    'alert-modal': AlertModal
  },
  computed: {
    ...mapState(['show_loading_mask']),
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
  position: absolute;
  top: 0px;
  width: 100%;
  font-size: $component-font-size;
  color: $component-font-color;

  #app-content {
    width: $app-window-width;
    margin: $header-height auto;
    padding-top: $app-container-padding-top;
    padding-bottom: $app-container-padding-bottom;
  }
  #loading-whole-mask {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100vh;
    z-index: 1000;
    background-color: rgba(0, 0, 0, 0.7);
    #loader-container{
      display: flex;
      flex-wrap: wrap;
      width: 11%;
      height: 10%;
      position: relative;
      top: 45%;
      left: 45%;
      padding-left: 2px;
      padding-right: 2px;
      .loader {
        position: relative;
        top: 20%;
        width: calc(20% - 8px);
        height: 80%;
        margin-left: 4px;
        margin-right: 4px;
        background-color: white;
      }
      .loader:nth-child(1) {
        animation: 0.75s 0.0s linear infinite both loading;
      }
      .loader:nth-child(2) {
        animation: 0.75s 0.15s linear infinite both loading;
      }
      .loader:nth-child(3) {
        animation: 0.75s 0.3s linear infinite both loading;
      }
      .loader:nth-child(4) {
        animation: 0.75s 0.45s linear infinite both loading;
      }
      .loader:nth-child(5) {
        animation: 0.75s 0.6s linear infinite both loading;
      }
      .message {
        position: absolute;
        width: 100%;
        height: 20%;
        top: 110%;
        color: white;
        display: flex;
        justify-content: center;
        font-size: 0.7rem;
      }
      @keyframes loading {
        0% {
          top: 20%;
          height: 80%;
        }
        50% {
          top: 0%;
          height: 100%;
        }
        100% {
          top: 20%;
          height: 80%;
        }
      }
    }
  }
}
</style>
