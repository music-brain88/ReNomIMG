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
      <span v-else-if="model.isTraining()">{{ this.loss }}</span>
      <span v-else-if="model.isValidating()">Validating</span>
      <span v-else-if="model.isStopping()">Stopping</span>
      <span v-else-if="model.isWeightDownloading()">Weight Downloading</span>
    </div>
    <div id="bar-area">
      <span v-if="isTitle"></span>
      <div id="bar-background" v-else>
        <div id="bar-front"
          :style="getWidthOfBar"
          :class="[getColorClass(model), getBarClass]">
        </div>
      </div>
    </div>
    <div id="button-stop-area" v-if="!isTitle">
      <i class="fa fa-stop-circle-o" aria-hidden="true" @click="onStop"></i>
    </div>
  </div>
</template>

<script>
import {RUNNING_STATE} from '@/const.js'
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
    ...mapGetters([
      'getColorClass'
    ]),
    model_id: function () {
      if (this.model === undefined) {
        return '-'
      } else {
        return this.model.id
      }
    },
    getBarClass: function () {
      if (this.model.isValidating() || this.model.isStopping() || this.model.isWeightDownloading()) {
        return 'validating'
      } else {
        return 'training'
      }
    },
    getWidthOfBar: function () {
      if (this.model.isValidating() || this.model.isStopping() || this.model.isWeightDownloading()) {
        return {
          width: '20%'
        }
      } else {
        if (this.total_batch === 0) {
          return {
            width: 0 + '%'
          }
        } else {
          return {
            width: (this.current_batch / this.total_batch) * 100 + '%'
          }
        }
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
  align-items: center;
  width: 100%;
  height: calc(#{$progress-bar-height}*0.8);
  padding-left: $progress-bar-margin;
  padding-right: $progress-bar-margin;
  margin-bottom: 10px;
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
    width: 15%;
    height: 100%;
  }
  #loss-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 18%;
    height: 100%;
  }

  #bar-area {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 33%;
    height: 70%;
    #bar-background {
      width: 100%;
      height: calc(100% - #{$bar-margin}*2);
      background-color: lightgray; 
      #bar-front.training {
        position: relative;
        top: 0;
        left: 0;
        height: 100%;
        transition: width 300ms;
      }
      #bar-front.validating {
        position: relative;
        top: 0;
        left: 0;
        height: 100%;
        transition: width 300ms;
        animation: move-bar 1.5s;
        animation-iteration-count: infinite;
        animation-timing-function: linear;
        animation-fill-mode: both;
        animation-delay: 0.1s;
      }
      

      @keyframes move-bar {
        0% {
          transform: translateX(-50%) scaleX(0);
        } 
        20% {
          transform: translateX(0%) scaleX(1);
        } 
        80% {
          transform: translateX(400%) scaleX(1);
        }
        100% {
          transform: translateX(450%) scaleX(0);
        }
      }
    }
  }
  #button-stop-area {
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 131%;
    width: 9%;
    height: 100%;
    text-align: center;
    color: lightgray;
    i {
      cursor: pointer;
    }
    i:hover {
      color: gray;
    }
  }
}
</style>
