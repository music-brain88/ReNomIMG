<template>
  <div id="project-panel">
    <div class="outer-project-panel" v-on:click="goDetectionPage">
      <div class="project-panel">
        <div id="row1">
          <div id="project-name">

<!-- ///プロジェクト名/////////////// -->
            Test Project
<!-- //////////////////////////////// -->

          </div>
          <div id="icon" @click="toggleComment">
            <i class="fa fa-commenting-o" aria-hidden="true"></i>
            <i class="fa fa-commenting" aria-hidden="true"></i>
          </div>
        </div>
        <div id="row2">

<!-- ///最終更新日/////////////////// -->
          2017/11/23 12:22:22
<!-- //////////////////////////////// -->

        </div>
        <div id="row3">
          <div id="accuracy">

<!-- ///最高バリデーション精度/////// -->
            98%
<!-- //////////////////////////////// -->

          </div>
          <div id="session-count">
            <div id="up">

<!-- ///終了したセッション数///////// -->
              5 Done/
<!-- //////////////////////////////// -->

            </div>
            <div id="down">
  
<!-- ///作成されたセッション数/////// -->
              12 Session
<!-- //////////////////////////////// -->

            </div>
          </div>
        </div>
        <div id="row4">
          <div id="name">
            Best Varidation Score
          </div>
          <div id="icon">
            <i class="fa fa-trash-o" aria-hidden="true"></i>
            <i class="fa fa-trash" aria-hidden="true"></i>
          </div>
        </div>
      </div>   
    </div>
    <div id="comment-space" v-if="showCommentFlag">
      Show comment here.
    </div>
  </div>
</template>

<script>
  export default {
    name: "ProjectPanel",
    props: ["project"],
    data: function () {
      return {
        showCommentFlag:false
      }
    },
    methods: {
      goDetectionPage: function() {
        let self = this;
        // this.$store.commit("setProjectId", {
        //   "project_id": this.project["project_id"]
        // });
        // this.$store.dispatch("loadPastLearnings")
        //   .then(function(response){
        //     self.$router.push({ path: '/detection_page' });
        // });
      },
      toggleComment: function () {
        this.showCommentFlag = !this.showCommentFlag
      },
    }
  }
</script>

<style lang="scss">

  .outer-project-panel {
    height: 120px;
    width: 230px;
    margin-right: 20px;
    margin-top: 20px;
    padding: 0px 0px 0px 0px;
    box-sizing:border-box;

    .project-panel {
      color: #4d4d4d;
      height: 100%;
      padding: 5px 10px 5px 10px;
      border: 2px solid #eee;
      box-sizing:border-box;
      background-color: #ffffff;

      #row1 {
        width:100%;
        display: flex;
        justify-content: space-around;
        #project-name {
          width:90%;
          height:100%;
          font-weight: bold;
          font-size: 1.2rem;
        }
        #icon {
          height:100%;
          text-align: right;
          .fa-commenting-o {display:inline}
          .fa-commenting {display:none}
        }
        #icon:hover {
          .fa-commenting {display:inline}
          .fa-commenting-o {display:none}
        }
      }

      #row2 {
        font-size: 0.7rem;
      }

      #row3 {
        display:flex;
        align-items: center;
        #accuracy {
          width: 60%;
          font-size: 1.6rem;
          padding-left: 10px;
          font-weight: bold;
          color: #185ce7;
        }
        #session-count {
          width: 40%;
          #up {
            font-size: 0.8rem; 
          }
          #down {
            font-size: 0.8rem; 
            padding-left:3px;
          }
        }
      }

      #row4 {
        width:100%;
        display: flex;
        justify-content: space-around;
        #name {
          width:90%;
          font-size: 0.7rem; 
          color: #ababab;
        }
        #icon {
          .fa-trash-o {display:inline}
          .fa-trash {display:none}
        }
        #icon:hover {
          .fa-trash {display:inline}
          .fa-trash-o {display:none}
        }
      }
    }

    .project-panel:hover {
      border: 2px solid #2d3e50;
    }
  }
</style>
