<template>
  <div id="detection-page">
    <div class="detection-content">
      <div class="detection-dashboard-and-detail">
        <dashboard></dashboard>
        <model-detail></model-detail>
      </div>

      <div class="detection-page-sidebar">
        <model-list></model-list>
      </div>
    </div>

    <div class="detection-sample-and-tags">
      <div class="model-sample">
        <model-sample></model-sample>
      </div>

      <div class="tag-list">
        <tag-list></tag-list>
      </div>
    </div>

    <image-modal v-if='show_modal_image_sample'/>

    <add-model-modal v-if="$store.state.add_model_modal_show_flag"></add-model-modal>
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
import AddModelModal from './add_model_modal.vue'
import WeightDownloadingModal from './weight_downloading_modal.vue'
import ImageModal from './image_modal.vue'

export default {
  name: 'DetectionPage',
  components: {
    'model-list': ModelList,
    'dashboard': DashBoard,
    'model-detail': ModelDetail,
    'model-sample': ModelSample,
    'tag-list': TagList,
    'add-model-modal': AddModelModal,
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
  },
  mounted: function () {
    this.$store.dispatch('updateModels', {'project_id': 1})
  }
}
</script>

<style lang="scss" scoped>
#detection-page {
  $header-height: 44px;
  $page-margin-top: 22px;
  $page-margin-horizontal: 72px;

  $side-bar-width: 240px;
  $side-bar-margin: 24px;

  display: flex;
  display: -webkit-flex;
  flex-direction: column;
  -webkit-flex-direction: column;

  width: calc(100% - #{$page-margin-horizontal}*2);
  margin: 0 $page-margin-horizontal;

  .detection-content {
    display: flex;
    display: -webkit-flex;
    height: 784px;

    .detection-dashboard-and-detail {
      width: calc(100% - #{$side-bar-width});
      height: 100%;
    }
    .detection-page-sidebar {
      width: $side-bar-width;
      height: 100%;
      margin-left: $side-bar-margin;
    }
  }

  .detection-sample-and-tags {
    display: flex;
    display: -webkit-flex;
    margin-bottom: 44px;

    .model-sample {
      width: calc(100% - #{$side-bar-width});
    }

    .tag-list {
      width: $side-bar-width;
      margin-left: $side-bar-margin;
    }
  }

}
</style>
