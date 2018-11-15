<template>
  <div id="app">
    <app-header/>
    <div id="app-content">
      <div id="page-tab">
        <div id="train-tab" class="page-tab" @click="onTabClick('train')">
          Train
        </div>
        <div id="predict-tab" class="page-tab" @click="onTabClick('predict')">
          Predict
        </div>
        <div id="dataset-tab" class="page-tab" @click="onTabClick('dataset')">
          Dataset
        </div>
        <div id="debug-tab" class="page-tab" @click="onTabClick('debug')">
          DEBUG
        </div>
      </div>
      <div id="container">
        <transition name="fade">
          <router-view></router-view> 
        </transition>
      </div>
    </div>
    <slide-menu/>
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
    onTabClick: function (page_name) {
      this.init()
      if (page_name === 'train') {
        this.$router.push({path: '/'})
        this.setCurrentPage(PAGE_ID.TRAIN)
      } else if (page_name === 'predict') {
        this.$router.push({path: '/predict'})
        this.setCurrentPage(PAGE_ID.PREDICT)
      } else if (page_name === 'dataset') {
        this.$router.push({path: '/dataset'})
        this.setCurrentPage(PAGE_ID.DATASET)
      } else if (page_name === 'debug') {
        this.$router.push({path: '/debug'})
        this.setCurrentPage(PAGE_ID.DEBUG)
      } else {
        console.log(page_name + 'is not supported page name.')
      }
    }
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
#app {
  width: 100%;
  #app-content {
    width: $app-window-width;
    margin: 0 auto;
    // padding-top: $app-container-padding-top;
    padding-bottom: $app-container-padding-bottom;
    #page-tab {
      height: $tab-content-height;
      min-height: $header-min-height;
      width: calc(100% - #{$component-block-margin}*2);
      letter-spacing: -0.4em; // For removing gap between divs.
      // margin: $component-block-margin;
      margin-bottom: $component-block-margin;
      .page-tab {
        line-height: normal;
        letter-spacing: 0em; // Remove letter-spacing
        display: inline-block;
        height: 100%;
        width: $tab-content-width;
        vertical-align: middle;
        text-align: center;
        font-size: 100%;
      }
      #train-tab {
        background-color: red;
      }
      #predict-tab {
        background-color: orange;
      }
      #dataset-tab {
        background-color: yellow;
      }
      #debug-tab {
        background-color: white;
      }
    }
  }
}
</style>
