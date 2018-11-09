<template>
  <div id="filter-item">
    <div id="filter-add" class="item" v-if="isAddButton" @click="showModal({add_filter: true})">
      <div id="add">
        <i class="fa fa-plus" aria-hidden="true"></i>
      </div>
    </div>
    <div id="filter-content" class="item" v-else>
      <div id="delete">
      </div>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations } from 'vuex'

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
    ...mapMutations(['showModal'])
  }
}
</script>

<style lang='scss'>
  #filter-item {
    width: calc(100%/3 - #{$model-filter-margin}*2);
    height: calc(100%/3.5 - #{$model-filter-margin}*2);
    margin: $model-filter-margin;
    .item {
      width: 100%;
      height: 100%;
      background-color: lightgray;
      #delete {
        width: 20%;
        height: 100%;
        background-color: black;
      }
      #add {
        border-style: dashed;
        border-color: black;
        border-width: 1.2px;
        width: 100%;
        height: 100%;
        font-family: "FontAwesome";
        text-align: center;
      }
    }
  }
</style>
