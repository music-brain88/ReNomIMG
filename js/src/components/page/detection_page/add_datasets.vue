<template>
  <div>
      <div class="form-group row">
        <div class="col-md-6">
          <form  v-on:submit.prevent="register">
            <h5>Dataset Setting</h5>
            <div class="container">
              <div class="row justify-content-center space-top">
                <label class="col-sm-4 col-form-label text-right">Dataset Name</label>
                <div class="col-sm-8">
                  <input type="text" v-model='name' class="form-control" placeholder="Dataset Name">
                </div>
              </div>

              <div class="row justify-content-center space-top">
                <label class="col-sm-4 col-form-label text-right">Discription</label>
                <div class="col-sm-8">
                  <textarea class="form-control" rows="5"></textarea>
                </div>
              </div>

              <div class="row justify-content-center space-top">
                <label class="col-sm-4 col-form-label text-right">Ratio of training data</label>
                <div class="col-sm-8">
                  <div class="input-group">
                    <input v-model='ratio' type="number" class="form-control" min="0" max="100" placeholder="80">
                    <div class="input-group-append">
                      <span class="input-group-text">%</span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="row justify-content-center space-top">
                <label class="col-sm-4 col-form-label text-right">Arrange number of data</label>
                <div class="col-sm-8">
                  <input type='checkbox' class="form-control">後ほど修正
                </div>
              </div>

              <div class="modal-button-area space-top float-right">
                <button>Confirm</button>
              </div>

            </div>
          </form>
        </div>
        <div class="col-md-6">
          <form>
            <h5>Train</h5>

            <div class="row space-top">

              <div v-if='dataset_detail.length===0'>

                <nav>
                  <div class="nav nav-tabs" id="nav-tab" role="tablist">
                    <a class="nav-item nav-link disabled active" id="nav-number-tab" data-toggle="tab" role="tab" aria-controls="nav-number" aria-selected="true">Total number</a>
                    <a class="nav-item nav-link disabled">All </a>
                    <a class="nav-item nav-link disabled">Train </a>
                    <a class="nav-item nav-link disabled">Vallidation </a>
                  </div>
                </nav>

                <div class="tab-content" id="nav-tabContent">
                  <div class="tab-pane fade show active" id="nav-number" role="tabpanel">
                    <div class="row">
                      <div class="col-md-9 offset-md-1">

                        <div class="space-top">
                          <div class="progress">
                            <div class="progress-bar" role="progressbar" style="width: 25%;" aria-valuenow="25" aria-valuemin="0" aria-valuemax="100">25%</div>
                          </div>
                        </div>
                        <div class="space-top">
                          <div class="progress">
                            <div class="progress-bar" role="progressbar" style="width: 20%;" aria-valuenow="20" aria-valuemin="0" aria-valuemax="100">20%</div>
                          </div>
                        </div>
                        <div class="space-top">
                          <div class="progress">
                            <div class="progress-bar" role="progressbar" style="width: 10%;" aria-valuenow="10" aria-valuemin="0" aria-valuemax="100">10%</div>
                          </div>
                        </div>
                        <div class="space-top">
                          <div class="progress">
                            <div class="progress-bar" role="progressbar" style="width: 45%;" aria-valuenow="45" aria-valuemin="0" aria-valuemax="100">45%</div>
                          </div>
                        </div>

                      </div>
                    </div>
                  </div>

                </div>

              </div>
              <div v-else>

                <nav>
                  <div class="nav nav-tabs" id="nav-tab" role="tablist">
                    <a class="nav-item nav-link disabled active" id="nav-number-tab" data-toggle="tab" role="tab" aria-controls="nav-number" aria-selected="true">Total number</a>
                    <a class="nav-item nav-link disabled">All {{dataset_detail.total}}</a>
                    <a class="nav-item nav-link disabled">Train {{dataset_detail.train_num}}</a>
                    <a class="nav-item nav-link disabled">Vallidation {{dataset_detail.valid_num}}</a>
                  </div>
                </nav>

                <div class="tab-content" id="nav-tabContent">
                  <div class="tab-pane fade show active" id="nav-number" role="tabpanel">
                    <div class="row">
                      <div class="col-md-9 offset-md-1">

                        <p>{{dataset_detail.class_maps}}</p>
                        <div class="row" v-for="(val, key) in dataset_detail.class_maps">
                          <div class="col-md-5">
                            Class name {{key}} :
                          </div>
                          <div class="col-md-7">
                            <div class="progress">
                              <div class="progress-bar" role="progressbar" :style="'width:' + calc_percentage(val, dataset_detail.total)+'%;'" :aria-valuenow="calc_percentage(val, dataset_detail.total)" aria-valuemin="0" aria-valuemax="100">{{calc_percentage(val, dataset_detail.total)}}%</div>
                            </div>
                          </div>
                        </div>

                      </div>
                    </div>
                  </div>

                </div>

              </div>

            </div>

            <div class="modal-button-area space-top-last float-right">
              <button>Create</button>
              <button>Reset</button>
            </div>

          </form>
        </div>
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
      name: ''
    }
  },
  computed: {
    ...mapState(['dataset_detail']),
    load_dataset_detail: function () {
      return this.$store.state.dataset_detail
    }
  },
  methods: {
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
      let f = this.$store.dispatch('loadDatasetSplitDetail', {ratio, name})
      f.finally(() => {
        this.ratio = DEFAULT_RATIO
        this.name = ''
      })
    },
    calc_percentage: function (train, total) {
      // let value = (train / this.$store.state.dataset_detail_max_value) * 100
      let value = (train / total) * 100
      return Math.round(value, 3)
    }
  }
}
</script>

<style lang="scss" scoped>
@import '@/../node_modules/bootstrap/scss/bootstrap.scss';

form {
  background: #FFFFFF;
  border:none;
}

nav {
  background: #ffffff;
  border: none;

}

.space-top{
  margin-top: 5%;
}
.space-top-last{
  margin-top: 20%;
}
</style>
