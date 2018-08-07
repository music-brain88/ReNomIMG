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
                <label class="col-sm-6 col-form-label text-right">Ratio of training data</label>
                <div class="col-sm-6">
                  <div class="input-group">
                    <input v-model='ratio' type="number" class="form-control" min="0" max="100" placeholder="80">
                    <div class="input-group-append">
                      <span class="input-group-text">%</span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="modal-button-area space-top float-right">
                <button class="submit">Confirm</button>
              </div>

            </div>
          </form>
        </div>
        <div class="col-md-6">
          <form  v-on:submit.prevent="register">
            <h5>Detail</h5>
            <div class="container">
              <div class="row space-top">
                <div class="col-md-12">
                  <div v-if='dataset_detail.length===0'>

                    <div class="row">
                      <div class="col-md-4">
                        Number of Images
                      </div>
                      <div class="col-md-4">
                        Train
                      </div>
                      <div class="col-md-4">
                        Vallidation
                      </div>
                    </div>
                    <div class="row space-top">
                      <div class="col-md-4">
                        All
                      </div>
                      <div class="col-md-8">
                        <div class="progress">
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
                      <div class="col-md-4">
                        Number of Images
                      </div>
                      <div class="col-md-4">
                        Train {{dataset_detail.train_num}}
                      </div>
                      <div class="col-md-4">
                        Vallidation {{dataset_detail.valid_num}}
                      </div>
                    </div>

                    <div class="row space-top">
                      <div class="col-md-4">
                        All
                      </div>
                      <div class="col-md-8">
                        <div class="progress">
                          <div class="progress-bar train-color" role="progressbar" :style="'width:' + calc_percentage(dataset_detail.train_num, dataset_detail.total)+'%;'" aria-valuemin="0" aria-valuemax="100"></div>
                          <div class="progress-bar validation-color" role="progressbar" :style="'width:' + calc_percentage(dataset_detail.valid_num, dataset_detail.total)+'%;'" aria-valuemin="0" aria-valuemax="100"></div>
                        </div>
                      </div>
                    </div>
                    <div class="row space-top">
                      <div class="col-md-4">
                        <span>Total Number of Tag</span>
                      </div>
                    </div>

                    <div v-for="(val, key) in dataset_detail.class_maps" class="row space-top">

                      <div class="col-md-4">
                        Class name {{key}} :
                      </div>
                      <div class="col-md-8">
                        <div class="progress">
                          <div class="progress-bar train-color" role="progressbar" :style="'width:' + calc_percentage(val, dataset_detail.total)+'%;'" :aria-valuenow="calc_percentage(val, dataset_detail.total)" aria-valuemin="0" aria-valuemax="100">{{calc_percentage(val, dataset_detail.total)}}%</div>
                        </div>
                      </div>

                    </div>
                  </div>
                </div>
              </div>

            <div class="modal-button-area space-top-last float-right">
              <button class="submit">Save</button>
              <button>Cancel</button>
            </div>

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
    ...mapState(['dataset_detail', 'loading_flg']),
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

input[type=checkbox]{
  height: auto;
  width: auto;
  clip-path:none;
}

.space-top{
  margin-top: 5%;
}
.space-top-last{
  margin-top: 20%;
}
.loading-space{
  padding-top: 60%;
  margin-top: 100%;
}
.train-color{
  background-color: $train-color;
}
.validation-color{
  background-color: $validation-color;
}
.submit{
  background-color: $push-button;
  color:$font-color;
}
</style>
