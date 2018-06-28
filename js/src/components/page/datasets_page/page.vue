<template>
  <div id="dataset-def-page">
    <form>
      <div>Name: <input v-model='name' type='text' placeholder='Dataset name' /> </div>
      <div>Ratio of training data: <input v-model='ratio'  type='number' placeholder='80.0' style='width:100px' /> % </div>
      <button @click="register">Create</button>
    </form>

    <table>
      <tr v-for="def in dataset_defs" :key="def.id">
       <td> {{ def.id }} </td>
       <td> {{ def.name }} </td>
       <td> {{ def.ratio * 100 }}%</td>
       <td> {{ datestr(def.created).toLocaleString() }} </td>
      </tr>
    </table>
  </div>
</template>

<script>
import { mapState } from 'vuex'

const DEFAULT_RATIO = 80.0
export default {
  name: 'PredictionPage',
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
    ...mapState(['dataset_defs'])
  },
  methods: {
    datestr: function (d) {
      return new Date(d)
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
#prediction-page {
  display: flex;
  display: -webkit-flex;
  flex-direction: column;
  -webkit-flex-direction: column;

  width: 100%;
  padding-bottom: 64px;

  .model-detail-area {
    width: 100%;
  }

  .prediction-result-area {
    width: 100%;
  }
}
</style>

