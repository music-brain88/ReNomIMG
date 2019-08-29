<template>
  <div id="app">
    <rnc-header :product-name="ProductName" />

    <div id="app-content">
      <rnc-drawer-menu
        :menu-obj="MenuObj"
        :show-menu="getShowSlideMenu"
        @show-drawer-menu="showDrawerMenu"
      />
      <div class="paging-area">
        <transition name="fade">
          <router-view />
        </transition>
      </div>
    </div>

    <rnc-modal-alert v-if="show_alert_modal" />
    <rnc-modal-confirm v-if="show_confirm_modal" />
    <rnc-modal-togglemask
      :show-modal="show"
      @show-modal="closeToggleModal"
    >
      <template slot="modal-contents">
        <rnc-modal-both v-if="show_modal.add_both" />
        <!-- TODO: <rnc-modal-dataset v-if="show_modal.add_dataset" />-->
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
// TODO: import RncModalDataset from './components/Organisms/rnc-modal-dataset/rnc-modal-dataset'
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
    // TODO: 'rnc-modal-dataset': RncModalDataset,
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
      version: '2.3.0 beta',
      copyright: '©︎2019 GRID INC. ALL rights reserved.'
    }
  },
  computed: {
    ...mapState([
      'show_loading_mask',
      'show_modal',
      'current_page',
      'show_alert_modal',
      'show_confirm_modal'
    ]),
    ...mapGetters([
      'getShowSlideMenu'
    ]),

    currentPage: function () {
      return this.current_page
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
  mounted () {
    this.$nextTick(function () {
      if (this.$route.path === '/dataset') {
        this.setCurrentPage(1)
      } else if (this.$route.path === '/') {
        this.setCurrentPage(0)
      } else if (this.$route.path === '/predict') {
        this.setCurrentPage(2)
      }
    })
  },
  methods: {
    ...mapActions([
      'init'
    ]),
    ...mapMutations([
      'setCurrentPage',
      'showSlideMenu',
      'showModal'
    ]),

    showDrawerMenu: function (page_number) {
      if (page_number !== null && page_number !== undefined) {
        this.setCurrentPage(page_number)
      }
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
  user-select: text;
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
  font-size: $fs-regular;
  color: $black;
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
}
</style>
