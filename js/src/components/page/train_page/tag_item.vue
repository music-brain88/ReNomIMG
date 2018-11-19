<template>
  <div id="tag-item">
    <div id="tag-id">
      {{ tagId }}
    </div>
    <div id="tag-title">
      {{ tagName }}
    </div>
  </div>
</template>

<script>
import { mapGetters } from 'vuex'

export default {
  name: 'TagItem',
  props: {
    tagName: '',
    tagId: '',
  },
  components: {
  },
  computed: {
  },
  created: function () {

  },
  methods: {

  }
}
</script>

<style lang='scss'>
#tag-item {
  display: flex;
  align-items: center;
  width: calc(100% - 10px);
  height: 5%;
  margin: 5px;
}
</style>
