<template lang="html">
<div class="row wrap">
  <div class="col-md-12 footer-content">
    <div class="row">
      <div class="col-md-3">
        <div class="media-wrapper">
          <img :src="url">
          <div class="content-margin-top">
              <p>ReNom_IMG {{version}}</p>
          </div>
        </div>
      </div>
      <div class="9">

      </div>
    </div>
  </div>
  <div class="col-md-12 copyright align-middle">
    <p><small>&copy; 2018 GRID INC, ALL rights reserved.</small></p>
  </div>
</div>
</template>

<script>

export default {
  data () {
    return {
      version: 'ver.0.9',
      url: require('../../../static/img/renomlogo.jpg')
    }
  }
}
</script>

<style lang="scss" scoped>
.wrap{
  color: $font-color;
  width: 100%;
  height:200px;
  .footer-content{
    padding-left: 3%;
    margin-top: 15%;
    background: $footer-color;
  }
  .copyright{
    padding-left: 3%;
    background: $footer-copyright-color;
    bottom: 0;
  }
}
.content-margin-top{
  margin-top: $content-margin;
}
.media-wrapper{
  margin-top: 10%;
}

</style>
