<template>
  <component-frame :width-weight="2" :height-weight="9">
    <template slot="header-slot">
      <div id="model-list-title">
        Model List
      </div>
      <div id="model-add-button">
      <select class="sort-menu" v-on:change="setSortOrder">
        <option v-for="item in getSortTitle" :value="item">{{item}}</option>
      </select>
      </div>
    </template>
    <div id="model-list">
      <model-item v-for="model in getFilteredModelList" :model="model">
      </model-item>
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
  align-content: flex-start;
  flex-wrap: wrap;
  padding: $model-list-margin;
}
#component-header {
  #model-list-title {
    display: inline-block;
    width: 49%;
    height: 100%;
  }
  #model-add-button {
    display: inline-block;
    width: 49%;
    height: 100%;
  }
}
</style>
