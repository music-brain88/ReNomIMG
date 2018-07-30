<template lang="html">
<footer>
  <div class="row">
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
        <div class="col-md-9">
          <!--none -->
        </div>
      </div>
    </div>
  </div>
  <div class="row">
    <div class="col-md-12 copyright align-middle">
      <div class="copyright-text">
        <small>&copy; 2018 GRID INC, ALL rights reserved.</small>
      </div>
    </div>
  </div>
</footer>
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
footer {
  display: block;
  width: 100%;
  padding: 0;
  margin:0;
  background: $footer-color;
  .row {
    margin-left: 0px;
    margin-right: 0px;
  }
  .footer-content{
    padding-left: 3%;
    background: $footer-color;
  }
  .copyright{
    background: $footer-copyright-color;
    .copyright-text{
      margin-top: 3px;
      margin-bottom: 3px;
      padding-left: 3%;
    }
  }
}
.content-margin-top{
  margin-top: $content-margin;
}
.media-wrapper{
  margin-top: $content-margin;
}

</style>
