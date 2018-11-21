<template>
  <div id="modal-add-filter">
    Add Model Filter
    <div id="add-filter">
      <div id="item">
        <input type="text">
      </div>
      <div id="condition">
        <select v-model="condition">
          <option>>=</option>
          <option>==</option>
          <option><=</option>
        </select>
      </div>
      <div id="value">
        <input type="text">
      </div>
      <div id="add">
        <input type="button" value="+">
      </div>
    </div>
    <div id="filter-list">
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
      'getCurrentTask',
      'getAlgorithmList',
      'getAlgorithmParamList',
      'getAlgorithmIdFromTitle'
    ]),
  },
  data: function () {
    return {
      item: '',
      condition: '',
      threshold: '',
    }
  },
  created: function () {

  },
  methods: {
    ...mapMutations(['addFilter']),
    creatrFilter: function () {
      const filter = new Filter()
      this.addFilter(this.item, this.condition, this.threshold)
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
