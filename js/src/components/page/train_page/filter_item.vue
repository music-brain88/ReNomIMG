<template>
  <div id="filter-item">
    <div id="filter-add" class="item" v-if="isAddButton">
      <div id="add">
        <i class="fas fa-plus"></i>
      </div>
    </div>
    <div id="filter-content" class="item" v-else>
      <div id="delete">
      </div>
    </div>
  </div>
</template>

<script>
import {mapGetters} from 'vuex'

export default {
  name: 'FilterItem',
  components: {
  },
  props: {
    isAddButton: {
      type: Boolean,
      default: false,
    },
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
  #filter-item {
    width: calc(100%/3 - #{$model-filter-margin}*2);
    height: calc(100%/3.5 - #{$model-filter-margin}*2);
    margin: $model-filter-margin;
    background-color: red;
    .item {
      width: 100%;
      height: 100%;
      #delete {
        width: 20%;
        height: 100%;
        background-color: black;
      }
      #add {
        width: 20%;
        height: 100%;
        font-family: "FontAwesome";
      }
    }
  }
</style>
