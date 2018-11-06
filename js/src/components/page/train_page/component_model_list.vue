<template>
  <component-frame :width-weight="2" :height-weight="9">
    <template slot="header-slot">
      Model List
    </template>
    <div id="model-list">
      <model-item/>
      <model-item/>
      <model-item/>
      <model-item/>
      <model-item/>
    </div>
  </component-frame>
</template>

<script>
import {mapGetters} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import ModelItem from '@/components/page/train_page/model_item.vue'

export default {
  name: 'ComponentModelList',
  components: {
    'component-frame': ComponentFrame,
    'model-item': ModelItem
  },
  computed: {

  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
#model-list {
  display: flex; 
  width: 100%;
  align-content: flex-start;
  flex-wrap: wrap;
  padding: $model-list-margin;
}
</style>
