<template>
  <component-frame :width-weight="6" :height-weight="3">
    <template slot="header-slot">
      Model Detail
    </template>
    <div id="model-detail">
      <div class="col">
        <div class="item" v-for="param in model_params_col1">
          <div class="item-title">{{ param.title }} :</div>
          <div class="item-content">{{ param.content }}</div>
        </div>
      </div>
      <div class="col">
        <div class="item" v-for="param in model_params_col2">
          <div class="item-title">{{ param.title }} :</div>
          <div class="item-content">{{ param.content }}</div>
        </div>
      </div>
    </div>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'

export default {
  name: 'ComponentModelDetail',
  components: {
    'component-frame': ComponentFrame
  },
  props: {
    model: Object
  },
  computed: {
    model_params_col1: function () {
      if (this.model === undefined) {
        return [
          {'title': 'Model ID', 'content': '-'},
          {'title': 'Algorithm', 'content': '-'}
        ]
      } else {
        // TODO: Return reformatted data.
        return this.model.id
      }
    },
    model_params_col2: function () {
      if (this.model === undefined) {
        return [
          {'title': 'Image Size', 'content': '- x -'},
          {'title': 'Batch Size', 'content': '-'},
          {'title': 'Total Epoch', 'content': '-'},
          {'title': 'Best Epoch', 'content': '-'},
          {'title': 'Best mAP', 'content': '-'},
          {'title': 'Best IOU', 'content': '-'}
        ]
      } else {
        // TODO: Return reformatted data.
        return this.model.id
      }
    }
  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
#model-detail {
  width: 100%;
  padding: $model-detail-margin;
  display: flex;
  .col {
    width: 50%;
    .item {
      width: 100%;
      display: flex;
      margin-bottom: $model-detail-item-margin-bottom;
      .item-title {
        width: 45%;
        color: $model-detail-item-title-font-color;
      }
      .item-content {
        width: 70%;
      }
    }
  }
}
</style>
