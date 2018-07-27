<template lang="html">
<footer>
  <div class="row">
    <div class="col-md-3">
      <div class="media-wrapper">
        <img :src="url">
      </div>
    </div>
    <div class="9">

    </div>
  </div>
</footer>
</template>

<script>

export default {
  data () {
    return {
      url: require('../../../static/img/renomlogo.jpg')
    }
  }
}
</script>

<style lang="scss" scoped>
footer{
  background: #1e264b;
}
</style>
