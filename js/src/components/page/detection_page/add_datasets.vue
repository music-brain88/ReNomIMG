<template>
  <div id="data-setting">
      <div class="form-group row">
        <div class="col-md-6 col-padding-clear">
          <form  >
            <h5>Dataset Setting</h5>
            <div class="container">
              <div class="row justify-content-center space-top">
                <label class="col-sm-5 col-form-label label">Dataset Name</label>
                <div class="col-sm-7">
                  <input type="text" v-model='name' class="form-control input-text sort-line" placeholder="Dataset Name">
                </div>
              </div>

              <div class="row justify-content-center space-top">
                <label class="col-sm-5 col-form-label label">Discription</label>
                <div class="col-sm-7">
                  <textarea v-model='discription' class="form-control sort-line" rows="7"></textarea>
                </div>
              </div>

              <div class="row justify-content-center space-top">
                <label class="col-sm-5 col-form-label label">Ratio of training data</label>
                <div class="col-sm-7">
                  <div class="input-group">
                    <input v-model='ratio' type="number" class="form-control input-text sort-line" min="0" max="100" placeholder="80">
                    <div class="input-group-append">
                      <span class="input-group-text input-text sort-line">%</span>
                    </div>
                  </div>
                </div>
              </div>


            </div>
          </form>
        </div>
        <div class="col-md-6 col-padding-clear">
          <form  v-on:submit.prevent="register">
            <h5>Detail</h5>
            <div class="container">
              <div class="row space-top">
                <div class="col-md-12">
                  <div v-if='dataset_detail.length===0'>

                    <div class="row">
                      <div class="col-md-6">
                        Number of Images
                      </div>
                      <div class="col-md-3 figure">
                        Train
                      </div>
                      <div class="col-md-3 figure">
                        Valid
                      </div>
                    </div>
                    <div class="row space-top">
                      <div class="col-md-6">
                        All
                      </div>
                      <div class="col-md-6">
                        <div class="progress figure total-progress">
                          <div class="progress-bar" role="progressbar" style="width:0%;" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                      </div>
                    </div>
                    <div class="row">
                      <div class="col-md-2 offset-5">
                        <div class="loading-space">
                          <div v-if='loading_flg' class="spinner-donut primary"></div>
                        </div>
                      </div>
                    </div>

                  </div>
                  <div v-else>

                    <div class="row">
                      <div class="col-md-6 col-form-label">
                        Number of Images
                      </div>
                      <div class="col-md-3 figure col-form-label">
                        Train {{dataset_detail.train_image_num}}
                      </div>
                      <div class="col-md-3 figure col-form-label">
                        Valid {{dataset_detail.valid_image_num}}
                      </div>
                    </div>

                    <div class="row space-top">
                      <div class="row col-md-12">
                        <div class="col-md-6 col-form-label">
                          All {{dataset_detail.total}}
                        </div>
                        <div class="col-md-6 col-form-label">
                          <div class="progress total-progress sort-line">
                            <div class="progress-bar train-color" role="progressbar" :style="'width:' + calc_percentage(dataset_detail.train_image_num, dataset_detail.total)+'%;'" aria-valuemin="0" aria-valuemax="100"></div>
                            <div class="progress-bar validation-color" role="progressbar" :style="'width:' + calc_percentage(dataset_detail.valid_image_num, dataset_detail.total)+'%;'" aria-valuemin="0" aria-valuemax="100"></div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <div class="row space-top-2nd">
                      <div class="col-md-6 col-form-label">
                        <span>Total Number of Tag</span>
                      </div>
                      <div class="col-md-6 col-form-label">
                        <span>{{calcTotaltag_num(dataset_detail.class_tag_list)}}</span>
                      </div>
                    </div>

                    <!-- tag preview -->
                   <div class="row space-top taglist-preview">

                      <div v-for=" data in dataset_detail.class_tag_list" class="row col-md-12">
                        <div class="col-md-6 col-form-label">
                          {{data.tags}} :
                        </div>
                        <div class="col-md-6 figure" @mouseenter="show_tag_data" @mouseleave="hidden_tag_data"  data-toggle="tooltip" data-placement="top" :title="data.train +'・'+ data.valid">
                          <div v-bind:class="{ 'tag-visible': show_tag_data_flg==true, 'tag-hidden': show_tag_data_flg==false }">{{data.train}}・{{data.valid}}</div>
                          <div class="progress figure tag-progress">
                            <div class="progress-bar train-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.train, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                            <div class="progress-bar validation-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.valid, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                          </div>
                        </div>
                      </div>
                      <div v-for=" data in dataset_detail.class_tag_list" class="row col-md-12">
                        <div class="col-md-6 col-form-label">
                          {{data.tags}} :
                        </div>
                        <div class="col-md-6 figure" @mouseenter="show_tag_data" @mouseleave="hidden_tag_data"  data-toggle="tooltip" data-placement="top" :title="data.train +'・'+ data.valid">
                          <div v-bind:class="{ 'tag-visible': show_tag_data_flg==true, 'tag-hidden': show_tag_data_flg==false }">{{data.train}}・{{data.valid}}</div>
                          <div class="progress figure tag-progress">
                            <div class="progress-bar train-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.train, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                            <div class="progress-bar validation-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.valid, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                          </div>
                        </div>
                      </div>

                      <div v-for=" data in dataset_detail.class_tag_list" class="row col-md-12">
                        <div class="col-md-6 col-form-label">
                          {{data.tags}} :
                        </div>
                        <div class="col-md-6 figure" @mouseenter="show_tag_data" @mouseleave="hidden_tag_data"  data-toggle="tooltip" data-placement="top" :title="data.train +'・'+ data.valid">
                          <div v-bind:class="{ 'tag-visible': show_tag_data_flg==true, 'tag-hidden': show_tag_data_flg==false }">{{data.train}}・{{data.valid}}</div>
                          <div class="progress figure tag-progress">
                            <div class="progress-bar train-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.train, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                            <div class="progress-bar validation-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.valid, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                          </div>
                        </div>
                      </div>
                      <div v-for=" data in dataset_detail.class_tag_list" class="row col-md-12">
                        <div class="col-md-6 col-form-label">
                          {{data.tags}} :
                        </div>
                        <div class="col-md-6 figure" @mouseenter="show_tag_data" @mouseleave="hidden_tag_data"  data-toggle="tooltip" data-placement="top" :title="data.train +'・'+ data.valid">
                          <div v-bind:class="{ 'tag-visible': show_tag_data_flg==true, 'tag-hidden': show_tag_data_flg==false }">{{data.train}}・{{data.valid}}</div>
                          <div class="progress figure tag-progress">
                            <div class="progress-bar train-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.train, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                            <div class="progress-bar validation-color"
                                role="progressbar" :style="'width:' + calc_percentage(data.valid, calc_max_tag_num(dataset_detail.class_tag_list))+'%;'"
                                aria-valuemin="0"
                                aria-valuemax="100">
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                    <!-- tag preview -->
 
                  </div>
                </div>
              </div>

            </div>
          </form>
        </div>
      </div>
      <div class="modal-button-area-confirm">
        <button @click="confirm" class="submit">Confirm</button>
      </div>
      <div v-if='dataset_detail.length!==0' class="modal-button-area">
        <button class="button" @click="hideAddModelModal">Cancel</button>
        <button class="submit"  @click="register">Save</button>
      </div>

  </div>
</template>

<script>
import { mapState } from 'vuex'
const DEFAULT_RATIO = 80.0

export default {
  data: function () {
    return {
      ratio: DEFAULT_RATIO,
      discription: '',
      name: '',
      id: '',
      show_tag_data_flg: false
    }
  },
  computed: {
    ...mapState(['dataset_detail', 'loading_flg']),
    load_dataset_detail: function () {
      return this.$store.state.dataset_detail
    }
  },
  methods: {
    hideAddModelModal: function () {
      this.$store.commit('setAddModelModalShowFlag', {'add_model_modal_show_flag': false})
    },
    register: function () {
      console.log('register')
      const name = this.name.trim()
      if (!name) {
        return
      }

      const ratio = parseFloat(this.ratio) / 100
      if ((ratio <= 0) || (ratio > 100)) {
        return
      }
      const u_id = this.id
      const discription = this.discription
      let f = this.$store.dispatch('registerDatasetDef', {ratio, name, u_id, discription})
      f.finally(() => {
        this.ratio = DEFAULT_RATIO
        this.discription = ''
        this.name = ''
        this.id = ''
      })
    },
    confirm: function () {
      console.log('confirm')
      const name = this.name.trim()
      if (!name) {
        return
      }

      const ratio = parseFloat(this.ratio) / 100
      if ((ratio <= 0) || (ratio > 100)) {
        return
      }

      const delete_id = this.id

      let u_id = this.gen_unique_id()
      this.id = u_id
      let discription = this.discription
      this.$store.dispatch('loadDatasetSplitDetail', {ratio, name, u_id, discription, delete_id})
      // f.finally(() => {
      //   this.ratio = DEFAULT_RATIO
      //  this.name = ''
      // })
    },
    calcTotaltag_num: function (list) {
      let tag_sum = 0
      for (let i in list) {
        tag_sum += list[i].train
        tag_sum += list[i].valid
      }
      return tag_sum
    },
    calc_percentage: function (target, total) {
      // let value = (train / this.$store.state.dataset_detail_max_value) * 100
      let value = (target / total) * 100
      return Math.round(value, 3)
    },
    show_tag_data: function () {
      this.show_tag_data_flg = true
      return this.show_tag_data_flg
    },
    hidden_tag_data: function () {
      this.show_tag_data_flg = false
      return this.show_tag_data_flg
    },
    gen_unique_id: function (encript_strong) {
      let default_strong = 1000
      if (encript_strong) {
        default_strong = encript_strong
      }
      let u_id = new Date().getTime().toString(16) + Math.floor(default_strong * Math.random()).toString(16)

      return u_id
    },
    calc_max_tag_num: function (taglist) {
      let max_value = 0
      for (let i in taglist) {
        let conpare = taglist[i].valid + taglist[i].train
        console.log(conpare)
        if (max_value < conpare) {
          max_value = conpare
        }
      }
      console.log('result:', max_value)
      return max_value
    }
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';

#data-setting{

  font-family: $content-inner-box-font-family;
  font-size: $content-inner-box-font-size;
  color:$font-color-label;
  $modal-content-padding: 32px;
  ::-webkit-input-placeholder {
    color: #999999;
  }

  // scroll setting

  div::-webkit-scrollbar{
    width: 6px;
  }
  div::-webkit-scrollbar-track{
    background: $body-color;
    border: none;
    border-radius: 6px;
  }
  div::-webkit-scrollbar-thumb{
    background: #aaa;
    border-radius: 6px;
    box-shadow: none;
  }


  h5{
    font-family: $content-inner-header-font-family;
    font-size: $content-inner-header-font-size;
  }
  form {
    background: #FFFFFF;
    border:none;
  }

  nav {
    background: #ffffff;
    border: none;

  }

  input[type=checkbox]{
    height: auto;
    width: auto;
    clip-path:none;
  }

  .form-control{
    border-radius: 0;
    font-size: $content-inner-box-font-size;
  }

  .label{
    padding-left: calc(#{$content-inner-box-font-size}*2);
  }

  .input-text{
    height: 20px;
  }
  .sort-line{
    margin-top: 10px;
  }

  .input-group-text{
    border-radius: 0;
  }

  .space-top{
    margin-top: 2%;
  }
  .space-top-2nd{
    margin-top: 5%;
  }

  .space-top-last{
    margin-top: 20%;
  }
  .loading-space{
    padding-top: 60%;
    margin-top: 100%;
  }

  .progress{
    border-radius: 0;
    background-color: $content-bg-color;
  }
  .total-progress{
    height:9px;
  }
  .tag-progress{
    height:6px;
  }


  .train-color{
    background-color: $train-color;
  }
  .validation-color{
    background-color: $validation-color;
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

  .modal-button-area-confirm {
    display: flex;
    flex-direction: row-reverse;
    position: absolute;
    bottom: calc(#{$modal-content-padding} - 10px);
    left:calc(403px - calc(0.5rem + 0.75rem + 15px));

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


  .col-padding-clear{
    padding: 0;
  }
  .figure{
    font-size: calc(#{$tab-figure-font-size}*0.8);
  }
  .tag-visible{
    visibility: visible;
  }
  .tag-hidden{
    visibility: hidden;
  }
  .taglist-preview{
    overflow-y:scroll;
    height:190px;
  }

}
</style>
