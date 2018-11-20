<template>
  <div id="modal-add-filter">
    Add Model Filter
    <div id="add-filter">
      <div id="item">
      </div>
      <div id="condition">
      </div>
      <div id="value">
      </div>
    </div>
    <div id="filter-list">
    </div>
  </div>
</template>

<script>
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
    }
  },
  created: function () {

  },
  methods: {
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
