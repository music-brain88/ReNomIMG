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
            Dataset
          </div>

          <div class="param-item">
            <div class="label">Dataset Name</div>
            <div class="item">
              <select v-model="dataset_def_id">
                <option v-for="d in dataset_defs" v-bind:value="d.id">
                  {{ d.name }}
                </option>
              </select>
            </div>
            <div class="input-alert" v-if="dataset_defs.length === 0">At least one dataset is needed</div>
          </div>
          <hr>
          <div class="sub-param-title">
            Algorithm Setting
          </div>

          <div class="param-item">
            <div class="label">CNN Architecture</div>
            <div class="item">
              <select class="algorithm-select-box" v-model="algorithm">
                <option value="0">YOLOv1</option>
                <option value="1">YOLOv2</option>
              </select>
            </div>
          </div>

          <div class="param-item">
            <div class="label">Train Whole Network</div>
            <div class="item">
              <select class="algorithm-select-box" v-model="train_whole_flag">
                <option value="0">False</option>
                <option value="1">True</option>
              </select>
            </div>
          </div>

          <div v-if="algorithm == 0" class="param-item">
            <div class="label">Cells</div>
            <div class="item">
              <input type="text" v-model="cells" maxlength="2">
            </div>
            <div class="input-alert" v-if="cells < 3">Cells must greater than 3</div>
            <div class="input-alert" v-if="cells > 20">Cells must lower than 20</div>
          </div>

          <div v-if="algorithm == 0" class="param-item">
            <div class="label">Bounding Box</div>
            <div class="item">
              <input type="text" v-model="bounding_box" maxlength="2">
            </div>
            <div class="input-alert" v-if="bounding_box < 1">Bounding Box must greater than 1</div>
            <div class="input-alert" v-if="bounding_box > 10">Bounding Box must lower than 10</div>
          </div>

          <div v-if="algorithm == 1" class="param-item">
            <div class="label">Anchors</div>
            <div class="item">
              <input type="text" v-model="anchor" maxlength="2">
            </div>
            <div class="input-alert" v-if="anchor < 1">Anchors must greater than 1</div>
            <div class="input-alert" v-if="anchor > 10">Anchors must lower than 10</div>
          </div>

        </div>




        <div class="sub-param-area">
          <div class="sub-param-title">
            Hyper Params
          </div>

          <div class="param-item">
            <div class="label">Image Width</div>
            <div v-if="algorithm == 0" class="item">
              <input type="text" v-model="image_width" maxlength="4">
              <div class="input-alert" v-if="image_width < 32">Image Width must greater than 32</div>
              <div class="input-alert" v-if="image_width > 1024">Image Width must lower than 1024</div>
            </div>
            <div v-if="algorithm == 1" class="param-item">
              <div class="item">
                <input type="text" v-model="image_height" maxlength="4" readonly="readonly">
              </div>
            </div>
          </div>

          <div class="param-item">
            <div class="label">Image Height</div>
            <div v-if="algorithm == 0" class="item">
              <input type="text" v-model="image_height" maxlength="4">
              <div class="input-alert" v-if="image_height < 32">Image Height must greater than 32</div>
              <div class="input-alert" v-if="image_height > 1024">Image Height must lower than 1024</div>
            </div>
            <div v-if="algorithm == 1" class="item">
              <input type="text" v-model="image_height" maxlength="4" readonly="readonly">
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
              <input type="text" v-model="total_epoch" maxlength="4">
            </div>
            <div class="input-alert" v-if="total_epoch < 1">Epoch must greater than 1</div>
            <div class="input-alert" v-if="total_epoch > 1000">Epoch must lower than 1000</div>
          </div>

          <div class="param-item">
            <div class="label">Batch Size</div>
            <div class="item">
              <input type="text" v-model="batch_size" maxlength="5">
            </div>
            <div class="input-alert" v-if="batch_size < 1">Batch Size must greater than 1</div>
            <div class="input-alert" v-if="batch_size > 512">Batch Size must lower than 512</div>
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
        <button @click="hideAddModelModal">Cancel</button>
        <button @click="runModel" :disabled="!isRunnable">{{ status }}</button>
      </div>

    </div>
  </div>
</template>

<script>
export default {
  name: 'AddModelModal',
  data: function () {
    return {
      dataset_def_id: 1,
      algorithm: 0,
      train_whole_flag: 0,
      total_epoch: 100,
      seed: 0,

      image_width: 448,
      image_height: 448,
      previous_image_width: 448,
      previous_image_height: 448,
      batch_size: 64,

      // YOLO params
      cells: 7,
      bounding_box: 2,

      // YOLO2 params
      anchor: 5
    }
  },
  computed: {
    status: function () {
      return this.$store.getters.getModelsFromState(1).length < 2 ? 'Run' : 'Reserve'
    },
    dataset_defs: function () {
      return this.$store.state.dataset_defs
    },
    isRunnable: function () {
      if (this.cells < 3 || this.cells > 20 ||
         this.bounding_box < 0 || this.bounding_box > 10 ||
         this.image_width < 32 || this.image_width > 1024 ||
         this.image_height < 32 || this.image_height > 1024 ||
         this.total_epoch < 0 || this.total_epoch > 1000 ||
         this.batch_size < 0 || this.batch_size > 512) {
        return false
      }
      return true
    }
  },
  watch: {
    algorithm (value) {
      if (parseInt(value) === 1) {
        this.previous_image_height = this.image_height
        this.previous_image_width = this.image_width
        this.image_height = 320
        this.image_width = 320
      } else {
        this.image_height = this.previous_image_height
        this.image_width = this.previous_image_width
      }
    }
  },
  methods: {
    hideAddModelModal: function () {
      this.$store.commit('setAddModelModalShowFlag', {'add_model_modal_show_flag': false})
    },
    runModel: function () {
      const hyper_parameters = {
        'total_epoch': parseInt(this.total_epoch),
        'batch_size': parseInt(this.batch_size),
        'seed': parseInt(this.seed),
        'image_width': parseInt(this.image_width),
        'image_height': parseInt(this.image_height),
        'train_whole_network': parseInt(this.train_whole_flag)
      }

      let algorithm_params = {}
      if (parseInt(this.algorithm) === 0) {
        algorithm_params = {
          'cells': parseInt(this.cells),
          'bounding_box': parseInt(this.bounding_box)
        }
      } if (parseInt(this.algorithm) === 1) {
        algorithm_params = {
          'anchor': parseInt(this.anchor)
        }
      }
      this.$store.dispatch('runModel', {
        dataset_def_id: parseInt(this.dataset_def_id),
        'hyper_parameters': hyper_parameters,
        'algorithm': this.algorithm,
        'algorithm_params': algorithm_params
      })

      this.hideAddModelModal()
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
  width: 100vw;
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
          position: relative;
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
          .input-alert {
            position: absolute;
            top: 44px;
            left: 132px;
            pading: 4px 8px;
            font-size: 12px;
            color: #ff0000;
          }
        }
      }
      hr {
        margin-top: 30px;
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

