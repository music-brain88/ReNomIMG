<template>
  <div id="model-progress">
    <div class="value-item">
      <div class="label">
        Model ID
      </div>
      <div class="value">
        {{ spacePadding(model.model_id, 6) }}
      </div>
    </div>

    <div class="value-item">
      <div class="label">
        Epoch
      </div>
      <div class="value">
        {{spacePadding(model.validation_loss_list.length)}}/{{spacePadding(model.hyper_parameters['total_epoch'], 4)}}
      </div>
    </div>

    <div class="value-item">
      <div class="label">
        Batch
      </div>
      <div class="value">
        {{spacePadding(current_batch, 3)}}/{{spacePadding(total_batch, 3)}}
      </div>
    </div>

    <div class="progress-bar-area">
      <div class="progress-bar-back">
        <div class="progress-bar">
          <div class="progress-bar-animation"
            v-bind:style="{backgroundColor: getColor(model.state, model.algorithm)}"
            v-bind:class="{animated: status===0}">
          </div>
        </div>
      </div>
    </div>

    <div class="value-item" v-if="status===0">
      <div class="label">
        Train Loss
      </div>
      <div class="value">
        {{spacePadding(train_loss, 7)}}
      </div>
    </div>

    <div class="value-item" v-if="status===3">
      <div class="label">
        Starting...
      </div>
      <div class="value" style='display: flex; align-items: center;'>
        <i class="fa fa-spinner fa-spin" aria-hidden="true"></i>
      </div>
    </div>

    <div class="value-item" v-if="status===4">
      <div class="label">
        Stopping...
      </div>
      <div class="value" style='display: flex; align-items: center;'>
        <i class="fa fa-spinner fa-spin" aria-hidden="true"></i>
      </div>
    </div>

    <div class="value-item" v-if="status===1">
      <div class="label">
        Validating
      </div>
      <div class="value" style='display: flex; align-items: center;'>
        <i class="fa fa-spinner fa-spin" aria-hidden="true"></i>
      </div>
    </div>

    <div class="stop-button-area">
      <div class="stop-button" @click="stopModel" v-if="status!==4">
        <i class="fa fa-pause-circle-o" aria-hidden="true"></i>
      </div>
    </div>
  </div>
</template>

<script>
var TRAIN = 0
var VALID = 1
var TRAIN_STARTING = 3
var TRAIN_STOPPING = 4

export default {
  name: "modelProgress",
  props: {
    "index": {
      type: Number,
      require: true
    },
    "model": {
      type: Object,
      require: true
    },
    "currentInfo": {
      type: Object,
      require: true
    }
  },
  data: function () {
    return {
      train_loss: "-",
      total_batch: "-",
      current_batch: "-",
      status: TRAIN_STARTING,
    }
  },
  created: function () {
    this.train_loss = "-"
    this.total_batch = "-"
    this.current_batch = "-"
    this.status = TRAIN_STARTING
    // this.$store.dispatch('getRunningModelInfo', {'model_id': this.model.model_id});
  },
  watch: {
    currentInfo: function(newVal) {
      this.updateProgressBar(newVal);
    }
  },
  updated: function() {
    this.updateProgressBar(this.currentInfo);
  },
  beforeDestroy: function () {
    this.status = TRAIN
  },
  methods: {
    updateProgressBar: function(info) {
      let epoch = this.model.epochs;
      if(!(typeof (info) == "string" || info instanceof String || this.model.state===TRAIN_STOPPING)) {
        const progress_bar_width = document.getElementsByClassName('progress-bar-back')[0].clientWidth;
        const current_width = info.current_batch / info.total_batch * progress_bar_width;
        let e = document.getElementsByClassName("progress-bar");

        if ('train_loss' in info){
          this.train_loss = info.train_loss.toFixed(3)
          this.status = info.status
          this.current_batch = info.current_batch
          this.total_batch = info.total_batch
        } else {
          this.train_loss = "-"
          this.total_batch = "-"
          this.current_batch = "-"
          this.status = TRAIN_STARTING
        }
        e[this.index].style.width = current_width + "px";
      }
    },
    stopModel: function() {
      if(confirm("学習を停止しますか？")){
        let self = this
        this.status = TRAIN_STOPPING
        this.$store.commit('setModelStopFlag', {
          "model_id": this.model.model_id,
        });
        this.$store.dispatch('stopModel', {
          "model_id": this.model.model_id,
        });
      }
    },
    getColor: function(model_state, algorithm) {
      return this.$store.getters.getColorByStateAndAlgorithm(model_state, algorithm);
    },
    spacePadding: function (s, count) {
      return (Array(count).join("\u0020") + new String(s)).substr(-(count))
    }
  }
}
</script>

<style lang="scss" scoped>
#model-progress {
  $content-margin: 8px;

  $label-color: #666666;
  $label-font-size: 12px;
  $value-font-size: 14px;

  $progress-bar-width: 24%;
  $progress-bar-height: 12px;
  $progress-bar-bg-color: #e8e8e8;
  $progress-bar-color: #000099;

  $stop-button-color: #999999;


  display: flex;
  display: -webkit-flex;

  margin-top: $content-margin;

  .value-item {
    display: flex;
    display: -webkit-flex;
    flex-direction: column;
    -webkit-flex-direction: column;

    margin-right: 12px;

    .label {
      color: $label-color;
      font-size: $label-font-size;
      line-height: $label-font-size;
    }

    .value {
      font-size: $value-font-size;
      line-height: $value-font-size;
    }
  }

  .stop-button-area {
    position: relative;
    .stop-button {
      position: absolute;
      bottom: 0;

      margin-right: $content-margin;
      line-height: $value-font-size;
      font-size: $value-font-size;
      color: $stop-button-color;
    }
  }

  .progress-bar-area {
    position: relative;
    width: $progress-bar-width;
    height: calc(#{$label-font-size} + #{$value-font-size});
    margin-right: $content-margin;

    .progress-bar-back {
      position: absolute;
      bottom: 2px;
      width: 100%;
      height: $progress-bar-height;
      background-color: $progress-bar-bg-color;
      border-radius: 1px;
      border-width: 1px;
      border-color: #a5a5a5;
      border-style: solid;
    }

    .progress-bar {
      position: absolute;
      height: 10px;
      .progress-bar-animation {
        border-radius: 1px;
        background-color: $progress-bar-color;
        height: 100%;
      }
      .animated{
        animation: AnimationName 3s ease infinite;
      }
      :not(.animated){
        width: 100%;
      }
      @keyframes AnimationName {
        0%{width: 0;}
        50%{width: 100%;}
        100%{width: 100%;}
      }
    }
  }
}
</style>
