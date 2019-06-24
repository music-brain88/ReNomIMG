<template>
  <div
    id="tag-item"
    :style="'height: ' + itemHeight + 'px;'"
  >
    <div
      id="tag-id"
      :style="{backgroundColor: tagColor}"
    >
      {{ tagId }}
    </div>
    <div id="tag-title">
      {{ tagName }}
    </div>
  </div>
</template>

<script>
import { getTagColor } from './../../../utils.js'

export default {
  name: 'RncTagListItem',
  props: {
    tagName: {
      type: String,
      default: 'Undefined'
    },
    tagId: {
      type: Number,
      default: 0
    },
    itemHeight: {
      type: Number,
      default: 28
    }
  },
  computed: {
    tagColor () {
      const color = getTagColor(this.tagId)
      return color
    }
  }
}
</script>

<style lang='scss'>
@import './../../../../static/css/unified.scss';

#tag-item {
  display: flex;
  align-items: center;
  width: 100%;
  border-bottom: solid 1px lightgray;
  #tag-id {
    width: 15%;
    height: 60%;
    color: white;
    text-align: center;
    margin-right: 10px;
    font-size: 0.7rem;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  #tag-title {
    width: 85%;
    color: gray;
    font-size: 0.8rem;
  }
}
</style>
