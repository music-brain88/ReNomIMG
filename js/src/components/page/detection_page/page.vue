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
    <image-modal v-if='show_modal_image_sample'/>

    <!-- <add-model-modal v-if="$store.state.add_model_modal_show_flag"></add-model-modal> -->
    <setting-page v-if="$store.state.add_model_modal_show_flag"></setting-page>
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
// import AddModelModal from './add_model_modal.vue'
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
  }
}
</script>

<style lang="scss" scoped>
 #detection-page {
  display: -webkit-flex;
  flex-direction: column;
  -webkit-flex-direction: column;

  margin: 0;
//  margin-top: $component-margin-top;
  width: 100%;

 }

</style>
