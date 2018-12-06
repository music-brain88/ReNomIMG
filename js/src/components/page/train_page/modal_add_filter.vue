<template>
  <div id="modal-add-filter">
    <div id="model-filter-title">
      Add Model Filter
    </div>
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
      <div id="add" @click="creatrFilter" :disabled="isDisabled" tabindex="0">
        <i class="fa fa-plus" aria-hidden="true"></i>
      </div>
    </div>
    <div id="filter-list">
      <div v-for="filterItem in getFilterList" id="filter-item">
        <div v-if="filterItem.item.type === 'select'" class="select-item">
          <div id="item">
            {{ filterItem.item.title }}
          </div>
          <div id="condition">
             == 
          </div>
          <div id="threshold">
            {{ filterItem.threshold.title }}
          </div>
          <div id="remove" @click="rmFilter(filterItem)">
            <i class="fa fa-times" aria-hidden="true"></i>
          </div>
        </div>
        <div v-if="filterItem.item.type === 'condition'" class="condition-item">
          <div id="item">
            {{ filterItem.item.title }}
          </div>
          <div id="condition">
            {{ filterItem.condition }}
          </div>
          <div id="threshold">
            {{ filterItem.threshold }}
          </div>
          <div id="remove" @click="rmFilter(filterItem)">
            <i class="fa fa-times" aria-hidden="true"></i>
          </div>
        </div>
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
    isDisabled: function () {
      return this.threshold === ''
    }
  },
  data: function () {
    return {
      condition: '==',
      threshold: '',
      itemObject: {
        item: 'select',
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
    ...mapMutations([
      'addFilter',
      'rmFilter'
    ]),
    creatrFilter: function () {
      if (this.isDisabled) return
      const filter = new Filter(this.itemObject, this.condition, this.threshold)
      this.addFilter(filter)
      this.resetForm()
      this.itemObject = {
        title: 'select',
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
  #model-filter-title {
    height: 7%;
    color: $component-font-color-title;
  }
  #add-filter {
    display: flex;
    width: 100%;
    height: 5%;
    #item {
      width: 40%;
      height: 100%;
      *{
        width: 100%;
        height: 100%;
      }
    }
    #condition {
      width: 10%;
      height: 100%;
      margin-left: 5px;
      *{
        width: 100%;
        height: 100%;
      }
    }
    #value {
      width: 40%;
      height: 100%;
      margin-left: 5px;
      *{
        width: 100%;
        height: 100%;
      }
    }
    #add {
      width: 10%;
      height: 100%;
      margin-left: 5px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.1rem;
      cursor: pointer;
      color: #ccc;
      &:hover {
        color: #555;;
      }
    }
  }

  #filter-list {
    margin-top: 10px;
    width: 100%;
    height: calc(88% - 20px);
    overflow: auto;
    #filter-item {
      display: flex;
      width: 100%;
      height: 8%;
      .select-item {
        display: flex;
        width: 100%;
        height: 100%;
      }
     .condition-item {
        display: flex;
        width: 100%;
        height: 100%;
      }
      #item {
        width: 40%;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      #condition {
        width: 10%;
        margin-left: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      #threshold {
        width: 40%;
        margin-left: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      #remove {
        width: 10%;
        margin-left: 5px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #ccc;
        cursor: pointer;
        font-size: 1.1rem;
        &:hover {
          color: #555;;
        }
      }
    }
  }
}
</style>
