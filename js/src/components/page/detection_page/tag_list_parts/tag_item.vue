<template>
  <div id='tag-item'>
  <span> {{ name }} </span>
    <div id='box' v-bind:style="{backgroundColor: getColor}"><span>{{cls_id}}</span></div>
  </div>
</template>

<script>

export default {
  name: "TagItem",
  props: {
    cls_id: undefined,
    name: undefined,
  },
  computed: {
    getColor: function () {
      let color_list = ["#f19f36", "#53b05f", "#536cff", "#f86c8e"]
      let c = this.$props.cls_id%4
      return color_list[c]
    }
  }
}
</script>

<style lang="scss" scoped>

#tag-item {
  display: flex;
  border-bottom: 1px solid #c3c3c3;
  box-sizing:border-box;
  font-size: 0.8rem;
  color: #5a5a5a;
  padding-top: 2px;
  padding-bottom: 2px;
  #box {
    display: flex;
    margin-left: auto;
    margin-right: 8px;
    border-radius: 2px;
    align-self: center;
    height: 1rem;
    width: 25px;
    color: #ffffff;
    span {
      width: 100%;
      text-align: center;
      align-self: center;
    }
  }
}

</style>
