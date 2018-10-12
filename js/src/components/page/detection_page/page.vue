<template>
  <div id="detection-page">
    <div class="row">
      <div class="col-md-9 col-sm-12">
        <dashboard></dashboard>
        <model-detail></model-detail>
        <model-sample></model-sample>
        <tag-list></tag-list>
      </div>
      <div class="col-md-3 col-sm-12 list-parent">
        <model-list></model-list>	
      </div>
    </div>
    
    <transition name="fade">
      <image-modal v-if='show_modal_image_sample'></image-modal>
    </transition>
    
    <transition name="fade">
      <setting-page v-if="$store.state.add_model_modal_show_flag"></setting-page>
    </transition>
    
    <weight-downloading-modal v-if="$store.state.weight_downloading_modal"></weight-downloading-modal>
  </div>
</template>

<script>
import { mapState } from 'vuex'
import ModelList from './model_list.vue'
import DashBoard from './dashboard.vue'
import ModelDetail from './model_detail.vue'
import ModelSample from './model_sample.vue'
import TagList from './tag_list.vue'
import WeightDownloadingModal from './weight_downloading_modal.vue'
import ImageModal from './image_modal.vue'
import DataSettingPage from './setting_page.vue'

export default {
  name: 'DetectionPage',
  components: {
    'model-list': ModelList,
    'dashboard': DashBoard,
    'model-detail': ModelDetail,
    'model-sample': ModelSample,
    'tag-list': TagList,
    'setting-page': DataSettingPage,
    'weight-downloading-modal': WeightDownloadingModal,
    'image-modal': ImageModal
  },
  computed: {
    ...mapState([
      'show_modal_image_sample'
    ])
  },
  created: function () {
    this.$store.dispatch('initData', {'project_id': 1})
    this.$store.dispatch('loadDatasetDef')
    this.$store.dispatch('loadClassMap')
  }
}
</script>

<style lang="scss" scoped>
 #detection-page {
  display: -webkit-flex;
  flex-direction: column;
  -webkit-flex-direction: column;

  margin: 0;
  width: 100%;

 }


.fade-enter-active, .fade-leave-active {
  transition: opacity .2s;
}
.fade-enter, .fade-leave-to {
  opacity: 0;
}

</style>
