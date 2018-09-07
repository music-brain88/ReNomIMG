<template>
  <div id="dataset-def-page">
    <div class="row">
      <div class="col-md-12 col-sm-12">
        <div class="title">
          <div class="row">
            <div class="col-md-10">
              <div class="title-text">
                Dataset List
              </div>
            </div>
            <div class="col-md-2">
              <!-- Button trigger modal -->
              <div class="panel" @click="showAddModelModal">
                <i class="fa fa-angle-right icon-navgation"></i> Setting of Dataset
              </div>
            </div>
          </div>
        </div>

        <div class="content">
          <div class="row col-md-12 clear-padding">
            <div class="col data-area table-responsive border-range">
              <table class="table table-sm table-borderless">
                <thead>
                  <tr>
                    <th>ID&nbsp;&nbsp;</th>
                    <th>Dataset name&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
                    <th>Train&nbsp;&nbsp;&nbsp;&nbsp;</th>
                    <th>Valid&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</th>
                    <th>Create date</th>
                  </tr>
                </thead>
                <tbody class="scroll-controll" v-if="dataset_defs.length!==0">
                <tr class="dataset-row" v-bind:class="{'selected': index === number }" v-for="(def, number) in dataset_defs" :key="def.id" @click="selectDataset(number), set_num(def.train_imgs)">
                   <td> {{ set_id(def.id) }}</td>
                   <td> {{ get_dataset_name(def.name) }} </td>
                   <td> {{ set_num(def.train_imgs) }} </td>
                   <td> {{ set_num(def.valid_imgs.length) }}</td>
                   <td class="date" > {{ formatdate(datestr(def.created)) }} </td>
                  </tr>
                </tbody>
                <tbody v-else>
                  <tr>
                   <td colspan="4"> Dataset is empty. Please set the dataset</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <!--Dataset preview -->
            <div class="col preview">
              <h5 class="dataset-name view-title" v-if=" index ==='' ">None dataset selected</h5>
              <h5 class="dataset-name view-title selected" v-else>{{dataset_defs[index].name}}</h5>
              <div class="row justify-content-center space-top">
                <div class="col-sm-12 discription-bottom">
                  <label class="discription-label">Discription</label>

                  <textarea v-if="index !== ''" class="form-control sort-line discription-form" rows="3" v-model="dataset_defs[index].discription" readonly=readonly></textarea>
                  <textarea v-else class="form-control sort-line discription-form" rows="3" readonly=readonly>None</textarea>

                </div>
                <div v-if=" index!=='' " class="row col-md-12 space-label-bottom">
                  <div class="col-md-6 col-form-label">
                    Number of Images
                  </div>
                  <div class="col-md-3 figure col-form-label">
                    Train {{dataset_defs[index].train_imgs}}
                  </div>
                  <div class="col-md-3 figure col-form-label">
                    Valid {{ dataset_defs[index].valid_imgs.length }}
                  </div>
                </div>

                <div v-else class="row col-md-12 space-label-bottom">
                  <div class="col-md-6 col-form-label">
                    Number of Images
                  </div>
                  <div class="col-md-3 figure col-form-label">
                    Train
                  </div>
                  <div class="col-md-3 figure col-form-label">
                    Valid
                  </div>
                </div>


                <div v-if=" index !=='' " class="row col-md-12">
                  <div class="col-md-6">
                    All {{dataset_defs[index].train_imgs + dataset_defs[index].valid_imgs.length}}
                  </div>
                  <div class="col-md-6">
                    <div class="progress total-progress sort-line">
                      <div class="progress-bar train-color" 
                        role="progressbar" 
                        :style="'width:' + calc_percentage(dataset_defs[index].train_imgs, dataset_defs[index].train_imgs + dataset_defs[index].valid_imgs.length)+'%;'" 
                        aria-valuemin="0" aria-valuemax="100">
                      </div>
                      <div class="progress-bar validation-color" 
                        role="progressbar" 
                        :style="'width:' + calc_percentage(dataset_defs[index].valid_imgs.length, dataset_defs[index].train_imgs + dataset_defs[index].valid_imgs.length)+'%;'" 
                        aria-valuemin="0" 
                        aria-valuemax="100">
                      </div>
                    </div>
                   </div>
                </div>

              </div>
            </div>
            <!--Dataset preview -->

            <!--figure -->
            <div class="col">
              <div class ="row space-label-bottom view-area">
                
                <div class="col-md-6 col-form-label">
                  <span>Total Number of Tag</span>
                </div>
                <div class="col-md-6 col-form-label clear-padding">
                  <span v-if="index!==''"> {{calcTotaltag_num(dataset_defs[index].class_tag_list)}}</span>
                </div>
              </div>

              <!-- taglist -->
              <div v-if=" index===''" class="row" v-bind:class="{'tag-list-view':test >= 5}">
                <div class="col-md-12 col-form-label">
                  <span>Please click table row.<br />After click you can see detail here</span>
                </div>
                
                <!-- <div v-for="i in test" class="row col-md-12">
                 
                  
                  <div class="col-md-6 col-form-label">
                    <span>{{i}}</span>
                  </div>
                  <div class="col-md-6 figure" @mouseenter="show_tag_data" @mouseleave="hidden_tag_data">
                    <div v-bind:class="{ 'tag-visible': show_tag_data_flg==true, 'tag-hidden': show_tag_data_flg==false }">{{300}}・{{400}}</div>
                    <div class="progress figure tag-progress">
                      <div class="progress-bar train-color"
                        role="progressbar" :style="'width:' + calc_percentage(i*20, 100)+'%;'"
                        aria-valuemin="0"
                        aria-valuemax="100">
                      </div>
                      <div class="progress-bar validation-color"
                        role="progressbar" :style="'width:' + calc_percentage(i*30, 100)+'%;'"
                        aria-valuemin="0"
                        aria-valuemax="100">
                      </div>
                    </div>
                  </div>
                </div> -->

              </div>
             
              <div v-else class="row" v-bind:class="{'tag-list-view':dataset_defs[index].class_tag_list.length >= 5}">
                <div v-for="data in  dataset_defs[index].class_tag_list" class="row col-md-12">
                  
                  <div class="col-md-6 col-form-label">
                    <span>{{data.tags}}</span>
                  </div>
                  <div class="col-md-6 figure" @mouseenter="show_tag_data" @mouseleave="hidden_tag_data">
                    <div v-bind:class="{ 'tag-visible': show_tag_data_flg==true, 'tag-hidden': show_tag_data_flg==false }">{{data.train}}・{{data.valid}}</div>
                    <div class="progress figure tag-progress">
                      <div class="progress-bar train-color"
                        role="progressbar" :style="'width:' + calc_percentage(data.train, calc_max_tag_num(dataset_defs[index].class_tag_list))+'%;'"
                        aria-valuemin="0"
                        aria-valuemax="100">
                      </div>
                      <div class="progress-bar validation-color"
                        role="progressbar" :style="'width:' + calc_percentage(data.valid, calc_max_tag_num(dataset_defs[index].class_tag_list))+'%;'"
                        aria-valuemin="0"
                        aria-valuemax="100">
                      </div>
                    </div>
                  </div>
                </div>
              </div>
              <!-- taglist -->

            </div>
            <!--figure -->


          </div>

        </div>
      </div>
    </div>


    <!-- <dataset-creating-modal v-if="modal_show_flag"></dataset-creating-modal> -->
    <add-datasets-modal v-if="$store.state.add_model_modal_show_flag"></add-datasets-modal>
  </div>

</template>

<script>
import { mapState } from 'vuex'
import DatasetCreatingModal from './dataset_creating_modal.vue'
import AddDatasetsModal from './add_datasets_modal.vue'

const DEFAULT_RATIO = 80.0
export default {
  name: 'DatasetsPage',
  components: {
    'dataset-creating-modal': DatasetCreatingModal,
    'add-datasets-modal': AddDatasetsModal
  },
  data: function () {
    return {
      ratio: DEFAULT_RATIO,
      name: '',
      index: '',
      show_tag_data_flg: false,
      test: 100
    }
  },
  created: function () {
    this.$store.dispatch('loadDatasetDef')
  },
  computed: {
    ...mapState(['dataset_defs']),
    modal_show_flag: function () {
      return this.$store.state.dataset_creating_modal
    }
  },
  methods: {
    datestr: function (d) {
      return new Date(d)
    },

    showAddModelModal: function () {
      this.$store.commit('setAddModelModalShowFlag', {
        'add_model_modal_show_flag': true
      })
    },
    show_tag_data: function () {
      this.show_tag_data_flg = true
      return this.show_tag_data_flg
    },
    hidden_tag_data: function () {
      this.show_tag_data_flg = false
      return this.show_tag_data_flg
    },
    calc_percentage: function (target, total) {
      let value = (target / total) * 100
      return Math.round(value, 3)
    },
    formatdate: function (date) {
      let format = 'yyyy/MM/dd'
      format = format.replace(/yyyy/g, date.getFullYear())
      format = format.replace(/MM/g, ('0' + (date.getMonth() + 1)).slice(-2))
      format = format.replace(/dd/g, ('0' + date.getDate()).slice(-2))
      format = format.replace(/HH/g, ('0' + date.getHours()).slice(-2))
      format = format.replace(/mm/g, ('0' + date.getMinutes()).slice(-2))
      format = format.replace(/ss/g, ('0' + date.getSeconds()).slice(-2))
      format = format.replace(/SSS/g, ('00' + date.getMilliseconds()).slice(-3))
      return format.slice(2)
    },
    selectDataset: function (index) {
      this.index = index
    },
    calcTotaltag_num: function (list) {
      let tag_sum = 0
      for (let i in list) {
        tag_sum += list[i].train
        tag_sum += list[i].valid
      }
      return tag_sum
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
      return max_value
    },
    get_dataset_name: function (dataset_name) {
      if (dataset_name.length > 12) {
        return dataset_name.slice(0, 12) + '...'
      }
      return dataset_name
    },
    set_num: function (num) {
      if (String(num).length < 6) {
        let zero = '0'
        let times = 6 - String(num).length
        for (let i = 0; i < times; i++) {
          num = zero + num
        }
      }
      return num
    },
    set_id: function (num) {
      if (String(num).length < 3) {
        let zero = '0'
        let times = 3 - String(num).length
        for (let i = 0; i < times; i++) {
          num = zero + num
        }
      }
      return num
    }
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';
#dataset-def-page {
  display: flex;
  display: -webkit-flex;
  // height: 784px; //784
  flex-direction: column;
  -webkit-flex-direction: column;

  margin: 0;
  width: 100%;
  //padding-bottom: 64px;
  
  /********************scroll***********************/
  tbody::-webkit-scrollbar{
    width: 6px;
  }
  tbody::-webkit-scrollbar-track{
    background: $content-bg-color;
    border: none;
    border-radius: 6px;
  }
  tbody::-webkit-scrollbar-thumb{
    background: #aaa;
    border-radius: 6px;
    box-shadow: none;
  }


  div::-webkit-scrollbar{
    width: 6px;
  }
  div::-webkit-scrollbar-track{
    background: $content-bg-color;
    border: none;
    border-radius: 6px;
  }
  div::-webkit-scrollbar-thumb{
    background: #aaa;
    border-radius: 6px;
    box-shadow: none;
  }
  /********************scroll***********************/

  form {
    background: #FFFFFF;
    border:none;
  }
  table{
    border:none; 
    tbody{
      margin-top:10px;
      tr:hover{
        color:$table-hover-font-color;
        cursor:pointer;
      }
    }
  }

  thead, tbody{
    display:block;
    background: $content-bg-color;
  }

  th{
    border-top: none;
    border-left: none;
    line-height: $content-inner-header-font-size;
    font-family: $content-inner-box-font-family;
    font-size: calc(#{$content-inner-box-font-size});
    background: transparent;
  }
  td{
    background: transparent;
    border-left: none;
    font-family: $content-inner-box-font-family;
    font-size: calc(#{$content-inner-box-font-size});//$content-inner-box-font-size;
  } 

  .date{
    font-family: $content-inner-box-font-family;
    font-size: calc(#{$content-inner-box-font-size - 1pt});//$content-inner-box-font-size;
    vertical-align: middle; 
  }

  .data-area{
    border-right: 1px solid $content-taglist-tagbox-font-color;
  }

  .scroll-controll{ 
    height:460px;
    overflow-y: scroll;
  }

  .title {
    height:$content-top-header-hight;
    font-size: $content-top-header-font-size;
    font-family: $content-top-header-font-family;
    background:$header-color;
    color:$font-color;
    margin-top: $component-margin-top;
    .title-text{
      line-height: $content-top-header-hight;
      margin-left: $content-top-heder-horizonral-margin;
    }
    .panel{
      background-color: $panel-bg-color;
      text-align: center;
      line-height: $content-top-header-hight;
      &:hover{
        cursor:pointer;
        background-color: $panel-bg-color-hover;
      }
    }
  }

  .content {
    height:$content-prediction-height;
    margin-top: $content-top-margin;
    display: flex;
    display: -webkit-flex;
    font-family: $content-inner-box-font-family;
    font-size: $content-inner-box-font-size;
    padding: $content-top-padding $content-horizontal-padding $content-bottom-padding;
    background-color: $content-bg-color;
    border: 1px solid $content-border-color;
    h5{
      font-family: $content-inner-box-font-family;
      font-size: $content-inner-header-font-size;
    }

    .preview{
      font-family: $content-inner-box-font-family;
      font-size: $content-inner-box-font-size;
    }
  }
  
  .clear-padding{
    padding-left: 0;
    padding-right:0;
  }
  
  .discription-label{
    margin-left:15px;
  }

  .discription-form{
    margin-left:25px;
    width:90%;
  }
  
  .figure{
    font-size: calc(#{$tab-figure-font-size}*0.8);
  }
  
  .discription-bottom{
    margin-bottom: 10%;
  }

  .space-label-bottom{
    margin-bottom: 5%;
  }
  
  .progress{
    border-radius: 0;
    background:$content-bg-color;
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
  
  .tag-visible{
    visibility: visible;
  }
  .tag-hidden{
    visibility: hidden;
  }
  
  .selected{
    color:$table-selected-font-color;
  }

  .view-title{
    margin-top:40px;
  }

  .view-area{
    margin-top:64px; 
  }

  .tag-list-view{
    height:350px; //175
    overflow-y:scroll;
  }

  
}
</style>
