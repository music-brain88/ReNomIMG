<template>
  <div id="app">
    <rnc-header :product-name="ProductName" />

    <div id="app-content">
      <rnc-drawer-menu
        :menu-obj="MenuObj"
        :show-menu="getShowSlideMenu"
        @showDrawerMenu="showDrawerMenu"
      />
      <div class="paging-area">
        <transition name="fade">
          <router-view />
        </transition>
      </div>
    </div>

    <rnc-modal-alert v-if="$store.state.show_alert_modal" />
    <rnc-modal-confirm v-if="$store.state.show_confirm_modal" />
    <rnc-modal-togglemask
      :show-modal="show"
      @show-modal="closeToggleModal"
    >
      <template slot="modal-contents">
        <rnc-modal-both v-if="show_modal.add_both" />
        <rnc-modal-dataset v-if="show_modal.add_dataset" />
        <rnc-modal-filter v-if="show_modal.add_filter" />
        <rnc-modal-image v-if="show_modal.show_image" />
        <rnc-modal-prediction-image v-if="show_modal.show_prediction_image" />
      </template>
    </rnc-modal-togglemask>

    <rnc-loading-mask :show-loading-mask="show_loading_mask" />

    <rnc-footer
      :version="version"
      :copyright="copyright"
    />
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapActions, mapState } from 'vuex'

import RncHeader from './components/Organisms/rnc-header/rnc-header.vue'
import RncFooter from './components/Organisms/rnc-footer/rnc-footer.vue'
import RncDrawerMenu from './components/Molecules/rnc-drawer-menu/rnc-drawer-menu.vue'
import RncLoadingMask from './components/Atoms/rnc-loading-mask/rnc-loading-mask.vue'

import RncModalAlert from './components/Organisms/rnc-modal-alert/rnc-modal-alert'
import RncModalConfirm from './components/Organisms/rnc-modal-confirm/rnc-modal-confirm'

import RncModalTogglemask from './components/Atoms/rnc-modal-togglemask/rnc-modal-togglemask.vue'
import RncModalBoth from './components/Organisms/rnc-modal-both/rnc-modal-both'
import RncModalDataset from './components/Organisms/rnc-modal-dataset/rnc-modal-dataset'
import RncModalFilter from './components/Organisms/rnc-modal-filter/rnc-modal-filter'
import RncModalImage from './components/Organisms/rnc-modal-image/rnc-modal-image'
import RncModalPredictionImage from './components/Organisms/rnc-modal-prediction-image/rnc-modal-prediction-image'

export default {
  name: 'App',
  components: {
    'rnc-header': RncHeader,
    'rnc-footer': RncFooter,
    'rnc-drawer-menu': RncDrawerMenu,
    'rnc-modal-alert': RncModalAlert,
    'rnc-modal-confirm': RncModalConfirm,
    'rnc-modal-togglemask': RncModalTogglemask,
    'rnc-modal-both': RncModalBoth,
    'rnc-modal-dataset': RncModalDataset,
    'rnc-modal-filter': RncModalFilter,
    'rnc-modal-image': RncModalImage,
    'rnc-modal-prediction-image': RncModalPredictionImage,
    'rnc-loading-mask': RncLoadingMask
  },
  data () {
    return {
      ProductName: 'ReNomIMG',
      MenuObj: [
        {
          'title': 'train',
          'icon': 'fa-home'
        },
        {
          'title': 'dataset',
          'icon': 'fa-database'
        },
        {
          'title': 'predict',
          'icon': 'fa-area-chart'
        }
      ],
      version: '2.1.1 beta',
      copyright: '©︎2019 GRID INC. ALL rights reserved.'
    }
  },
  computed: {
    ...mapState(['show_loading_mask', 'show_modal']),
    ...mapGetters(['getShowSlideMenu']),
    currentPage: function () {
      return this.$store.state.current_page
    },
    show: function () {
      return Object.values(this.show_modal).some((d) => d)
    }
  },
  watch: {
    currentPage: function () {
      this.init()
    }
  },
  created: function () {
    this.init()
  },
  methods: {
    ...mapActions(['init']),
    ...mapMutations(['setCurrentPage', 'showSlideMenu', 'showModal']),
    showDrawerMenu: function (page_name) {
      this.showSlideMenu(false)
    },
    closeToggleModal: function () {
      this.showModal({ 'all': false })
    },
  }
}
</script>

<style lang='scss'>
@import './../static/css/unified.scss';

@font-face{
  font-family: $header-product-name-font-family;
  src: url("./../static/fonts/OpenSans-Regular.ttf");
}

@font-face{
  font-family: $component-header-font-family;
  src: url("./../static/fonts/OpenSans-Light.ttf");
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
  background-color: #f4f4f2;

  #app-content {
    width: $app-window-width;
    margin-top: $header-height;
    /*margin-bottom: 10px;*/
    padding-top: $app-container-padding-top;
    padding-bottom: $app-container-padding-bottom;
    .paging-area {
      width: 100%;
      max-width: $max-width;
      margin: 0 auto;
    }
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
