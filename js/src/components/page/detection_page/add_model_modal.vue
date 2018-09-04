<template>
  <div id="add-model">
      <div class="form-group row">
        <div class="col-md-6 col-padding-clear">
          <form>
            <div class="sub-param-title">
              <h5>Dataset</h5>
            </div>
            <div class="form-row justify-content-center space-top">
              <label class="col-sm-6 label col-form-label">
                Dataset Name
              </label>
              <div class="col-sm-6">
                <div class="row">
                  <div class="col-sm-12"> 
                    <select class="form-control sort-line" v-model="dataset_def_id">
                      <option v-for="d in dataset_defs" v-bind:value="d.id">
                        {{ d.name }}
                      </option>
                    </select>
                  </div>
                </div>
                <div class="row">
                  <div class="link col-sm-12 text-right">
                    <a class="link"> &gt; Setting of Dataset</a>
                  </div>
                </div>
              </div>
              
            </div>

            <div class="sub-param-title category-top">
              <h5>Algorithm</h5>
            </div>

            <div class="form-row justify-content-center space-top">

              <label class="col-sm-6 label col-form-label">
                CNN Architecture
              </label>
              <div class="col-sm-6 form-group">
                <select class="form-control sort-line" v-model="algorithm">
                  <option value="0">YOLOv1</option>
                  <option value="1">YOLOv2</option>
                </select>
              </div>
            </div>

            <div class="form-row justify-content-center space-top">

              <label class="col-sm-6 label col-form-label">
                Train Whole Network
              </label>
              <div class="col-sm-6">
                <select class="algorithm-select-box form-control sort-line" v-model="yolo1_train_whole_flag" v-if="algorithm == 0">
                  <option value="0">False</option>
                  <option value="1">True</option>
                </select>
                <select class="algorithm-select-box form-control sort-line" v-model="yolo2_train_whole_flag" v-if="algorithm == 1">
                  <option value="0">False</option>
                  <option value="1">True</option>
                </select>
              </div>
            </div>

            <div v-if="algorithm == 0" class="form-row justify-content-center space-top">

              <label class="col-sm-6 label col-form-label">
                Cells
              </label>
              <div class="col-sm-6">
                <input type="text" class="form-control sort-line" v-model="cells" maxlength="2">
                <div class="input-alert" v-if="cells < 3">Cells must greater than 3</div>
                <div class="input-alert" v-if="cells > 20">Cells must lower than 20</div>
              </div>
            </div>

            <div v-if="algorithm == 0" class="form-row justify-content-center space-top">

              <label class="col-sm-6 label col-form-label">
                Bounding Box
              </label>
              <div class="col-sm-6">
                <input type="text" class="form-control sort-line" v-model="bounding_box" maxlength="2">
                <div class="input-alert" v-if="bounding_box < 1">Cells must greater than 3</div>
                <div class="input-alert" v-if="bounding_box > 10">Cells must lower than 20</div>
              </div>
            </div>

            <div v-if="algorithm == 1" class="form-row justify-content-center space-top">

              <label class="col-sm-6 label col-form-label">
                Anchors
              </label>
              <div class="col-sm-6">
                <input type="text" class="form-control" v-model="anchor" maxlength="2">
                <div class="input-alert" v-if="anchor < 1">Anchors must greater than 1</div>
                <div class="input-alert" v-if="anchor > 10">Anchors must lower than 10</div>
              </div>
            </div>

          </form>
        </div>

        <div class="col-md-6 col-padding-clear">
          <form>
            <div class="sub-param-title">
              <h5>Hyper params</h5>
            </div>
            <div class="row justify-content-center space-top">
              <label class="col-sm-6 label col-form-label">
                Image Width
              </label>
              <div v-if="algorithm == 0" class="col-sm-6">
                <input type="text" class="form-control sort-line" v-model="image_width" maxlength="4">
                <div class="input-alert" v-if="image_width < 32">Image Width must greater than 32</div>
                <div class="input-alert" v-if="image_width > 1024">Image Width must lower than 1024</div>
              </div>
              <div v-if="algorithm == 1" class="col-sm-6">
                <input type="text" class="form-control sort-line"  v-model="image_width" maxlength="4" readonly="readonly">
              </div>
            </div>
          

            <div class="row justify-content-center space-top">
              <label class="col-sm-6 label col-from-label">
                Image Hight
              </label>
              <div v-if="algorithm == 0" class="col-sm-6">
                <input type="text" class="form-control sort-line" v-model="image_height" maxlength="4">
                <div class="input-alert" v-if="image_height < 32">Image Height must greater than 32</div>
                <div class="input-alert" v-if="image_height > 1024">Image Height must lower than 1024</div>
              </div>
              <div v-if="algorithm == 1" class="col-sm-7">
                <input type="text" class="form-control"  v-model="image_height" maxlength="4" readonly="readonly">
              </div>
            </div>


            <div class="sub-param-title training-loop">
              <h5>Training Loop Setting</h5>
            </div>

            <div class="row justify-content-center space-top">
              <label class="col-sm-6 label col-form-label">
                Total Epoch
              </label>
              <div v-if="algorithm == 0" class="col-sm-6">
                <input type="text" class="form-control sort-line is-invalid" v-bind:class="{'is-invalid': total_epoch < 1, 'is-invalid': total_epoch > 1000}" v-model="total_epoch" maxlength="4">
                <div class="input-alert text-danger" v-if="total_epoch < 1">Epoch must greater than 1</div>
                <div class="input-alert text-danger" v-if="total_epoch > 1000">Epoch must lower than 1000</div>
              </div>
              <div v-if="algorithm == 1" class="col-sm-7">
                <input type="text" class="form-control sort-line"  v-model="total_epoch" maxlength="4">
              </div>
            </div>
            
            <div class="row justify-content-center space-top">
              <label class="col-sm-6 label col-form-label">
                Batch Size
              </label>
              <div class="col-sm-6">
                <input type="text" class="form-control sort-line" v-model="yolo1_batch_size" maxlength="5" v-if="algorithm == 0">
                <input type="text" class="form-control sort-line" v-model="yolo2_batch_size" maxlength="5" v-if="algorithm == 1" readonly="readonly">
                <div class="input-alert text-danger" v-if="batch_size < 1">Batch Size must greater than 1</div>
                <div class="input-alert text-danger" v-if="batch_size > 512">Batch Size must lower than 512</div>
              </div>
            </div>

          </form>
        </div>

        <!-- <div class="param-item">
          <div class="label">Seed</div>
          <div class="item">
            <input type="text" v-model="seed" maxlength="5">
          </div>
        </div> -->
    </div>

    <div class="modal-button-area">
      <button class="button" @click="hideAddModelModal">Cancel</button>
      <button class="submit" @click="runModel" :disabled="!isRunnable">{{ status }}</button>
    </div>

  </div>
</template>

<script>

import { mapState } from 'vuex'

export default {
  name: 'AddModelModal',
  data: function () {
    return {
      dataset_def_id: 1,
      algorithm: 0,
      yolo1_train_whole_flag: 0,
      yolo2_train_whole_flag: 1,
      train_whole_flag: 0,
      total_epoch: 100,
      seed: 0,

      image_width: 448,
      image_height: 448,
      previous_image_width: 448,
      previous_image_height: 448,
      yolo1_batch_size: 64,
      yolo2_batch_size: 16,
      batch_size: 64,

      // YOLO params
      cells: 7,
      bounding_box: 2,

      // YOLO2 params
      anchor: 5
    }
  },
  computed: {
    ...mapState(['gpu_num']),

    status: function () {
      return this.$store.getters.getModelsFromState(1).length < this.gpu_num ? 'Run' : 'Reserve'
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
         this.batch_size < 0 || this.batch_size > 512 ||
         this.$store.state.dataset_defs.length === 0) {
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
      if (this.algorithm === 0) {
        this.batch_size = this.yolo1_batch_size
        this.train_whole_flag = this.yolo1_train_whole_flag
      } else {
        this.batch_size = this.yolo1_batch_size
        this.train_whole_flag = this.yolo1_train_whole_flag
      }

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
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';

$modal-color: #000000;
$modal-opacity: 0.7;

$modal-content-bg-color: #fefefe;
$modal-content-padding: 32px;

$modal-title-font-size: 24px;
$modal-sub-title-font-size: 16px;

$content-margin: 8px;
$content-label-width: 120px;

#add-model{
  font-family: $content-inner-box-font-family;
  font-size: $content-inner-box-font-size;
  color:$font-color-label;
    ::-webkit-input-placeholder {
    color: #999999;
  }
  .link{
    color:#006ea1;
    font-size:calc(#{$content-inner-box-font-size}*0.8);
  }

  h5{
    font-family:$content-inner-header-font-family;
    font-size:$content-inner-header-font-size;
  }
  
  form {
    background: #FFFFFF;
    border:none;
  }

  .form-control{
    font-size: calc(#{$content-inner-box-font-size} - 1pt);
    padding: 0;
    border-radius: 0;
    height: 20px;
    // line-height: 10px;
    }
  .label{
    padding-left: calc(#{$content-inner-box-font-size}*2);
  }
}

.modal-title {
  font-size: $content-inner-header-font-size;
}

 .sort-line{
    margin-top: 10px;
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
      font-size: $content-inner-header-font-size;
    } 

    .param-item {
      display: flex;
      position: relative;
      margin-top: $content-margin;

      // .label {
      //   width: $content-label-width;
      //   font-size: $content-inner-box-font-size;
      //   line-height: $content-inner-box-font-size*1.5;
      // }
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
        color:$content-setting-modal-error-color;
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
    bottom: calc(#{$modal-content-padding} - 10px);
    right: $modal-content-padding;
    .submit{
      font-size: $push-button-font-size;
      height:$push-button-size;
      width:88px;
      background-color: $push-button;
      color:$font-color;
      line-height: calc(#{$push-button-size}*0.4);
    }
    .button{
      font-size: $push-button-font-size;
      height:$push-button-size;
      width:88px;
      background-color:#FFFFFF;
      border: 1px solid $push-cancel;
      line-height: calc(#{$push-button-size}*0.4);
      margin-left:11px;
    }

  }
  .space-top {
    margin-top: 2%;
  }
  .category-top{
   margin-top:5%;
  }
  .training-loop{
    margin-top:3%;
  }
  .col-padding-clear{
   padding: 0;
  }
  .input-alert{
    font-size:calc(#{$content-inner-box-font-size}*0.8);
  }
</style>
