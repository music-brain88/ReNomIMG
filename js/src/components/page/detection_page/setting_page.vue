<template>
  <div id="add-model-modal">
    <div class="modal-background" @click="hideAddModelModal"></div>
      <div class="modal-content">
        <div class="modal-body">
          <nav>
            <div class="nav nav-tabs" id="nav-tab" role="tablist">
              <a class="nav-item nav-link" v-bind:class="{'active': tab_show_flag }" id="nav-model-tab" @click="changeTab(true)">Setting of New Model</a>
              <a class="nav-item nav-link" v-bind:class="{'active': !tab_show_flag }" id="nav-dataset-tab" @click="changeTab(false)">Setting of Dataset</a>
            </div>
          </nav>

          <div class="tab-content" id="nav-tabContent">
            <div v-if="tab_show_flag">
              <add-model-modal></add-model-modal>
            </div>
            <div v-else>
              <add-detasets></add-detasets>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import AddModelModal from './add_model_modal.vue'
import AddDetasets from './add_datasets.vue'
import jquery from 'jquery'
import bootstrap from 'bootstrap'

export default {
  components: {
    'add-model-modal': AddModelModal,
    'add-detasets': AddDetasets,
    'jquery': jquery,
    'bootstrap': bootstrap
  },
  computed: {
    tab_show_flag: function () {
      return this.$store.state.modal_tab_show_flag
    }
  },
  methods: {
    hideAddModelModal: function () {
      this.$store.commit('setAddModelModalShowFlag', {'add_model_modal_show_flag': false})
    },
    changeTab: function (changeflag) {
      this.$store.commit('setChangeModalTabShowFlag', {'modal_tab_show_flag': changeflag})
    }
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';
#add-model-modal {
  $modal-color: #000000;
  $modal-opacity: 0.7;

  $modal-content-bg-color: #FFFFFF;
  $modal-content-padding: 32px;

  $content-margin: 8px;

  position: fixed;
  left: 0;
  top: $application-header-hight;
  width: 100vw;
  height: calc(100vh - #{$application-header-hight});
  z-index: 3;

  a{
    opacity: 1;
    transition: opacity 0s;
  }

  #nav-model-tab{
    display: block;
    width:205px;
    height:35px;
    background-color: $tab-bg-color;
    color: $font-color;
    border-radius: 0px;
    text-align: center;
    &.active{
      background-color: #FFFFFF;
      color: #000000;
    }
    &:hover{
      cursor:pointer;
    }
  }
  #nav-dataset-tab{
    display: block;
    width:205px;
    height:35px;
    background-color: $tab-bg-color;
    color: $font-color;
    border-radius: 0px;
    text-align: center;
    &.active{
      background-color: #FFFFFF;
      color: #000000;
    }
    &:hover{
      cursor:pointer;
    }
  }

  nav {
    background: #ffffff;
    border: none;
    padding: 0;
  }
  @media (min-width: 576px){
    .modal-dialog{
      max-width: 70%;
    }
  }

  .modal-background {
    width: 100%;
    height: 100%;
    background-color: $modal-color;
    opacity: $modal-opacity;
  }
  .modal-content {
    display: flex;
    flex-direction: column;

    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);
    padding-left: $content-horizontal-padding;
    padding-right: $content-horizontal-padding;
    padding-top: $content-top-padding;
    padding-bottom: $content-bottom-padding;
    width: calc(#{$modal-content-width} * #{$max-width}); //896
    height:$modal-content-height;
    background-color: $content-bg-color;
    opacity: 1;
    border-radius: 0;
  }
  .modal-body{
    background-color:$content-bg-color;
    padding: 0;
  }
}
</style>
