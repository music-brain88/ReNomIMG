<template>
  <div id="progress-bar">
    <div id="model-id-area">
      <span v-if="isTitle">Model</span>
      <span v-else>{{ this.model_id }}</span>
    </div>
    <div id="epoch-area">
      <span v-if="isTitle">Epoch</span>
      <span v-else>{{ this.current_epoch }} / {{ this.total_epoch }}</span>
    </div>
    <div id="batch-area">
      <span v-if="isTitle">Batch</span>
      <span v-else>{{ this.current_batch }} / {{ this.total_batch }}</span>
    </div>
    <div id="loss-area">
      <span v-if="isTitle">Loss</span>
      <span v-else>{{ this.loss }}</span>
    </div>
    <div id="bar-area">
      <span v-if="isTitle">Progress</span>
      <div id="bar-background" v-else>
      </div>
    </div>
    <div id="button-stop-area" v-if="!isTitle">
      <i class="fa fa-stop-circle-o" aria-hidden="true" @click="onStop"></i>
    </div>
  </div>
</template>

<script>
import { mapGetters, mapMutations, mapState, mapActions } from 'vuex'
export default {
  name: 'ProgressBar',
  props: {
    model: Object,
    isTitle: {
      type: Boolean,
      default: false
    }
  },
  data: function () {
    return {

    }
  },
  computed: {
    model_id: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.id
      }
    },
    current_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.nth_epoch
      }
    },
    current_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.nth_batch
      }
    },
    total_epoch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.total_epoch
      }
    },
    total_batch: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.total_batch
      }
    },
    loss: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.last_batch_loss.toFixed(3)
      }
    }
  },
  created: function () {

  },
  methods: {
    ...mapActions(['stopModelTrain']),
    onStop: function () {
      if (this.model) {
        this.stopModelTrain(this.model.id)
      }
    }
  }
}
</script>

<style lang='scss'>
#progress-bar {
  display: flex;
  flex-direction: row;
  width: 100%;
  height: $progress-bar-height;
  padding-left: $progress-bar-margin;
  padding-right: $progress-bar-margin;
  font-size: 80%;
  text-align: center;

  #model-id-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 12.5%;
    height: 100%;
  }
  #epoch-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 12.5%;
    height: 100%;
  }
  #batch-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 12.5%;
    height: 100%;
  }
  #loss-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 20%;
    height: 100%;
  }
  #bar-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 35.5%;
    height: 100%;
    #bar-background {
      width: 100%;
      height: calc(100% - #{$bar-margin}*2);
      background-color: gray; 
    }
  }
  #button-stop-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 7%;
    height: 100%;
    text-align: center;
    color: gray;
    i {
      cursor: pointer;
    }
    i:hover {
      color: lightgray;
    }
  }
}
</style>
