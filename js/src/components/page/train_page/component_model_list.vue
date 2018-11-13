<template>
  <component-frame :width-weight="2" :height-weight="8">
    <template slot="header-slot">
      <div id="model-list-title">
        Model List
      </div>
      <select class="sort-menu" v-on:change="setSortOrder">
        <option v-for="item in getSortTitle" :value="item">{{item}}</option>
      </select>
    </template>
    <div id="model-list" class="scrollbar-container">
      <model-item :is-add-button="true"/>
      <model-item v-for="(model, index) in getFilteredModelList" :model="model"/>
    </div>
  </component-frame>
</template>

<script>
import { mapGetters, mapMutations} from 'vuex'
import ComponentFrame from '@/components/common/component_frame.vue'
import ModelItem from '@/components/page/train_page/model_item.vue'

export default {
  name: 'ComponentModelList',
  components: {
    'component-frame': ComponentFrame,
    'model-item': ModelItem
  },
  computed: {
    ...mapGetters(['getFilteredModelList', 'getSortTitle']),
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['setSortOrder'])
  }
}
</script>

<style lang='scss'>
#model-list {
  display: flex; 
  width: 100%;
  height: 100%;
  align-content: flex-start;
  flex-wrap: wrap;
  padding: $model-list-margin;
  overflow: visible scroll;
}
#component-header {
  display: flex; 
  justify-content:space-between;
  #model-list-title {
    display: inline-block;
    height: 100%;
  }
  select {
    margin-left: auto;
  }
}
</style>
