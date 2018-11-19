<template>
  <component-frame :width-weight="2" :height-weight="10"
    :style="{position: 'sticky', top: 10 + 'px'}" class="model-list">
    <template slot="header-slot">
      Model List
      <div id="add-button" @click="showModal({add_both: true})">
        <i class="fa fa-plus" aria-hidden="true"></i>&nbsp;New
      </div>
    </template>

    <div id="model-filter">
      Model Filter
      <i class="fa fa-ellipsis-h" aria-hidden="true"></i>
    </div>

    <div id="model-groupby">
      Group by
      <select>
        <option value="none">-</option>
        <option value="dataset">Dataset</option>
        <option value="algorithm">Algorithm</option>
      </select>
    </div>

    <div id="model-titles">
      <span class="title-row">
        <div class="title selected">ID
          <i class="fa fa-sort-desc" aria-hidden="true"></i>&nbsp;&nbsp;
        </div>
        <div class="title">Alg
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
      </span>
      <span class="title-row">
        <div class="title">Loss
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
        <div class="title">/ Recall
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
        <div class="title">/ Precision
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
        <div class="title">/ F1
          <i class="fa fa-sort-desc" aria-hidden="true"></i>
        </div>
      </span>
    </div>

    <div id="model-list" class="scrollbar-container">
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
    'model-item': ModelItem,
  },
  computed: {
    ...mapGetters(['getFilteredModelList', 'getSortTitle']),
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['setSortOrder', 'showModal']),
    addModel: function () {
    }
  }
}
</script>

<style lang='scss'>

#add-button {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 100%;
  width: 33%;
  background-color: $component-header-sub-color;
}

#model-filter,#model-groupby,#model-titles {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: calc(#{$model-filter-height}*0.8);
  width: 100%;
  padding: $model-list-margin;
  margin-bottom: $model-list-margin;
  color: gray;
  font-size: 90%;
  background-color: white;
}

#model-groupby {
  select {
    border: none;
    background: transparent;
    border: solid 1px #eeeeee;
    height: 100%;
    appearance: none;
    line-height: 120%;
  }
}

#model-titles {
  font-size: 50%;
  color: gray;
  display: block;

  .selected {
    color:  $component-header-sub-color;
    font-weight: bold;
  }

  .title-row {
    display: flex;
    width: 100%;
    height: 50%;
  }
  .title {
    height: 100%;
    margin-left: 3px;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  i {
    font-size: 150%;
    text-align: center;
    position: relative;
    top: -1.5px;
    margin-left: 3px;
  }
}

#model-list {
  display: flex; 
  align-content: flex-start;
  flex-wrap: wrap;
  width: 100%;
  height: 100%;
  overflow: visible scroll;
}

.model-list {
  #component-header {
    display: flex; 
    justify-content:space-between;
  }
}
.model-list#component-frame {
  #frame-content {
    background-color: transparent;
  }
}
</style>
