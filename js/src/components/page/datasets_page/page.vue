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
            <div class="col-md-4 data-area">
              <table class="table">
                <thead>
                  <tr>
                    <th>Dataset name</th>
                    <th>Train</th>
                    <th>Validation</th>
                    <th>Create date</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="def in dataset_defs" :key="def.id">
                   <td> {{ def.id }} </td>
                   <td> {{ def.name }} </td>
                   <td> {{ def.ratio * 100 }}%</td>
                   <td> {{ datestr(def.created).toLocaleString() }} </td>
                  </tr>
                </tbody>

              </table>
            </div>

            <!--Dataset preview -->
            <div class="col-md-4 preview">
              <!-- <form>
                <div>Name: <input v-model='name' type='text' placeholder='Dataset name' /> </div>
                <div>Ratio of training data: <input v-model='ratio'  type='number' placeholder='80.0' style='width:100px' /> % </div>
                <button @click="register">Create</button>
              </form> -->
              <h5>Dataset Setting</h5>
              <div class="row justify-content-center space-top">
                <div class="col-sm-12">
                  <label class="discliption">Discription</label>
                  <textarea class="form-control sort-line discliption" rows="3"></textarea>
                </div>
              </div>
            </div>
            <!--Dataset preview -->

            <!--figure -->
            <div class="col-md-4">
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
      name: ''
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

    register: function () {
      const name = this.name.trim()
      if (!name) {
        return
      }

      const ratio = parseFloat(this.ratio) / 100
      if ((ratio <= 0) || (ratio > 100)) {
        return
      }

      let f = this.$store.dispatch('registerDatasetDef', {ratio, name})
      f.finally(() => {
        this.ratio = DEFAULT_RATIO
        this.name = ''
      })
    }
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';
#dataset-def-page {
  display: flex;
  display: -webkit-flex;
  height: 784px;
  flex-direction: column;
  -webkit-flex-direction: column;

  margin: 0;
  margin-top: $component-margin-top;
  width: 100%;
  padding-bottom: 64px;

  form {
    background: #FFFFFF;
    border:none;
  }
  table{
    border:none;
  }
  th{
    border-top: none;
    border-left: none;
    background: $content-bg-color;
    line-height: $content-inner-header-font-size;
    font-family: $content-inner-box-font-family;
    font-size: calc(#{$content-inner-box-font-size});
  }
  td{
    border-left: none;
    font-family: $content-inner-box-font-family;
    font-size: calc(#{$content-inner-box-font-size})//$content-inner-box-font-size;
  }
  .data-area{
    border-right: 1px solid $content-taglist-tagbox-font-color;
    overflow-y: scroll;
  }

  .title {
    height:$content-top-header-hight;
    font-size: $content-top-header-font-size;
    font-family: $content-top-header-font-family;
    background:$header-color;
    color:$font-color;
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
  }
  .discliption{
    margin-left:15px;
  }
}
</style>
