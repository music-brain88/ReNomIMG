<template>
  <div id="modal-add-filter">
    Add Model Filter
    <div id="add-filter">
      <div id="item">
        <select v-model="itemObject" v-on:change="resetForm">
          <option v-for="item of getFilterItemsOfCurrentTask" :value="item">{{item.title}}</option>
        </select>
      </div>
      <div id="condition">
        <select id="variable-condition" v-model="condition" v-if="itemObject.type !== 'select'">
          <option>>=</option>
          <option>==</option>
          <option><=</option>
        </select>
        <select id="fixed-condition" v-model="condition" v-else disabled>
          <option selected>==</option>
        </select>
      </div>
      <div id="value">
        <input type="text" v-if="itemObject.type !== 'select'" v-model="threshold">
        <select v-else v-model="threshold">
          <option v-for="opt in itemObject.options" :value="opt">{{opt.title}}</option>
        </select>
      </div>
      <div id="add">
        <input type="button" value="+" @click="creatrFilter">
      </div>
    </div>
    <div id="filter-list">
      <div v-for="filterItem in getFilterList">
        {{ filterItem }}
      </div>
    </div>
  </div>
</template>

<script>
import Filter from '@/store/classes/filter.js'
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'

export default {
  name: 'ModalAddFilter',
  components: {
  },
  computed: {
    ...mapState([]),
    ...mapGetters([
      'getFilterItemsOfCurrentTask',
      'getFilterList',
    ]),
  },
  data: function () {
    return {
      condition: '==',
      threshold: '',
      itemObject: {
        type: 'condition',
        options: [],
        min: 0,
        max: 1,
      }
    }
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['addFilter']),
    creatrFilter: function () {
      const filter = new Filter(this.itemObject, this.condition, this.threshold)
      this.addFilter(filter)
      this.resetForm()
      this.itemObject = {
        type: 'condition',
        options: [],
        min: 0,
        max: 1,
      }
    },
    resetForm: function () {
      this.condition = '=='
      this.threshold = ''
    }
  }
}
</script>

<style lang='scss'>
#modal-add-filter {
  width: 50%;
  height: 100%;
  padding: 10px;
  #add-filter {
    display: flex;
    width: 100%;
    height: 5%;
    #item {
      width: 40%;
      height: 100%;
    }
    #condition {
      width: 20%;
      height: 100%;
    }
    #value {
      width: 40%;
      height: 100%;
    }
  }
}
</style>
