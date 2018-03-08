<template>
  <div id="add-model-modal">
    <div class="modal-background" @click="hideAddModelModal"></div>
    <div class="modal-content">
      <div class="modal-title">
        Setting of New Training Model
      </div>

      <div class="modal-param-area">
        <div class="sub-param-area">
          <div class="sub-param-title">
            Algorithm Setting
          </div>

          <div class="param-item">
            <div class="label">CNN Architecture</div>
            <div class="item">
              <select class="algorithm-select-box" v-model="algorithm">
                <option value="0">YOLO</option>
              </select>
            </div>
          </div>

          <div v-if="algorithm == 0" class="param-item">
            <div class="label">Cells</div>
            <div class="item">
              <input type="text" v-model="cells" maxlength="5">
            </div>
          </div>

          <div v-if="algorithm == 0" class="param-item">
            <div class="label">Bounding Box</div>
            <div class="item">
              <input type="text" v-model="bounding_box" maxlength="5">
            </div>
          </div>
        </div>

        <div class="sub-param-area">
          <div class="sub-param-title">
            Hyper Params
          </div>

          <div class="param-item">
            <div class="label">Image Width</div>
            <div class="item">
              <input type="text" v-model="image_width" maxlength="5">
            </div>
          </div>

          <div class="param-item">
            <div class="label">Image Height</div>
            <div class="item">
              <input type="text" v-model="image_height" maxlength="5">
            </div>
          </div>
        </div>

        <div class="sub-param-area">
          <div class="sub-param-title">
            Training Loop Setting
          </div>

          <div class="param-item">
            <div class="label">Total Epoch</div>
            <div class="item">
              <input type="text" v-model="total_epoch" maxlength="5">
            </div>
          </div>

          <div class="param-item">
            <div class="label">Batch Size</div>
            <div class="item">
              <input type="text" v-model="batch_size" maxlength="5">
            </div>
          </div>

          <!-- <div class="param-item">
            <div class="label">Seed</div>
            <div class="item">
              <input type="text" v-model="seed" maxlength="5">
            </div>
          </div> -->
        </div>
      </div>

      <div class="modal-button-area">
        <button @click="hideAddModelModal">キャンセル</button>
        <button @click="runModel">RUN</button>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  name: "AddModelModal",
  data: function() {
    return {
      algorithm: 0,
      total_epoch: 100,
      seed: 0,

      image_width: 448,
      image_height: 448,
      batch_size: 64,

      // YOLO params
      cells: 7,
      bounding_box: 2,
    }
  },
  methods: {
    hideAddModelModal: function() {
      this.$store.commit("setAddModelModalShowFlag", {"add_model_modal_show_flag": false});
    },
    runModel: function() {
      const self = this
      this.$store.dispatch("checkDatasetDir").then(function(success){
        if(!success)return;

        const hyper_parameters = {
          'total_epoch': self.total_epoch,
          'batch_size': self.batch_size,
          'seed': self.seed,
          'image_width': self.image_width,
          'image_height': self.image_height,
        }

        let algorithm_params = {}
        if(self.algorithm == 0) {
          algorithm_params = {
            "cells": self.cells,
            "bounding_box": self.bounding_box,
          }
        }
        self.$store.dispatch("runModel", {
          'hyper_parameters': hyper_parameters,
          'algorithm': self.algorithm,
          'algorithm_params': algorithm_params,
        });
      });

      this.hideAddModelModal();
    }
  }
}
</script>

<style lang="scss" scoped>
#add-model-modal {
  $app-max-width: 1280px;
  $header-height: 35px;

  $modal-color: #000000;
  $modal-opacity: 0.7;

  $modal-content-width: 80%;
  $modal-content-height: 70%;
  $modal-content-bg-color: #fefefe;
  $modal-content-padding: 32px;

  $modal-title-font-size: 24px;
  $modal-sub-title-font-size: 16px;

  $content-margin: 8px;
  $content-label-width: 120px;
  $content-font-size: 16px;

  position: fixed;
  left: 0;
  top: $header-height;
  width: 100%;
  height: calc(100vh - #{$header-height});

  .modal-background {
    width: 100%;
    height: 100%;
    background-color: $modal-color;
    opacity: $modal-opacity;
  }

  .modal-content {
    display: flex;
    flex-direction: column;

    position: absolute;
    top: 50%;
    left: 50%;
    -webkit-transform: translateY(-50%) translateX(-50%);
    transform: translateY(-50%) translateX(-50%);

    width: $modal-content-width;
    height: $modal-content-height;
    max-width: $app-max-width;
    padding: $modal-content-padding;
    background-color: $modal-content-bg-color;
    opacity: 1;

    .modal-title {
      font-size: $modal-title-font-size;
      font-weight: bold;
    }

    .modal-param-area {
      display: flex;

      .sub-param-area {
        flex-grow: 1;

        display: flex;
        flex-direction: column;

        margin: $content-margin;
        border-top: 2px solid #cccccc;

        .sub-param-title {
          margin-top: $content-margin;
          font-size: $modal-sub-title-font-size;
          font-weight: bold;
        }

        .param-item {
          display: flex;

          margin-top: $content-margin;

          .label {
            width: $content-label-width;
            font-weight: 500;
            font-size: $content-font-size;
            line-height: $content-font-size*1.5;
          }
          .item {
            margin-left: $content-margin;
            width: $content-label-width;
            input {
              width: 100%;
            }
          }
        }
      }
    }

    .modal-button-area {
      display: flex;
      flex-direction: row-reverse;

      position: absolute;
      bottom: $modal-content-padding;
      right: $modal-content-padding;
    }
  }
}
</style>

