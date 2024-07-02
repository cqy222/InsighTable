<!-- <template>
    <div id = 'refresh-dialog' @click="getFile">
    </div>
</template>

<script>
  import { mapState, mapMutations } from 'vuex'

  export default {
    name: 'RefreshDialog',
    components: {},
    data() {
      return {
       search: ' ',
       fileList: [],
       selectedTabularDataName: null,
       tempSelection: null,
       tabularDataObjList: null,
      } 
    },
    watch: {
    },
    props: ['refreshDialogKey'],
    created: function () {
    },
    beforeMount: function() {
      if (this.refreshDialogKey == 1) {  // 将要显示的是RL结果弹窗 
        this.tabularDataObjList = sysDatasetObj.RLDataObjList
      }
      else {  // 将要显示的是普通表格文件弹窗
        this.tabularDataObjList = sysDatasetObj.tabularDataObjList
      }
    },
    mounted: function() {
        // this.search = ""
        // this.selectedTabularDataName = this.initTreeDataName
        // this.setCurrent("Console Sales.xlsx")
    },
    computed: {
        ...mapState([
        ])
    },
    watch: {
      refreshDialogKey: function (new_val) {
        if (new_val == 1) { // 将要显示的是RL结果弹窗 
          this.tabularDataObjList = sysDatasetObj.RLDataObjList
        }
        else { // 将要显示的是普通表格文件弹窗
          this.tabularDataObjList = sysDatasetObj.tabularDataObjList
        }
      },
    },
    methods: {
        dataTableRowDBClick: function() {
        },
        dataTableRowClick: function(row) {
            let fileName = row.filename
            this.tempSelection = fileName
            //  update the selected tabular dataset
            this.setCurrent(fileName)
        },
        closeDataDialog: function() {
            this.$bus.$emit('close-data-dialog', false)
        },
        confirmSelection: function() {
            // let self = this

            //  confirm the selected tabular dataset
            if (this.tempSelection != null) {
              // console.log("this.tempSelection", this.tempSelection)
              //   let selectionExisted = (this.treeDataArray.map(function(e) { return e.filename; })
              //     .indexOf(this.tempSelection) !== -1)
              //   console.log("selectionExisted", selectionExisted)
              //   if (selectionExisted) {
              //       this.selectedTabularDataName = this.tempSelection
              //       this.updateSelectedTabularDatasetName(this.selectedTabularDataName)
              //       this.tempSelection = null
              //       self.$cookies.set('selected-data-name', this.selectedTabularDataName)
              //   }
              this.selectedTabularDataName = this.tempSelection
              
              if (this.refreshDialogKey == 1) { // 选择rl数据
                this.updateSelectedRLDatasetName(this.selectedTabularDataName)
              }
              else {  // 选择普通数据
                this.updateSelectedTabularDatasetName(this.selectedTabularDataName)
              }
              
              this.tempSelection = null
              // this.$cookies.set('selected-data-name', this.selectedTabularDataName)
            }

            this.$bus.$emit('close-data-dialog', true)
        },
        getFile: function() {
            console.log('hhhhhhhhhhh')
            return axios
            .get("http://127.0.0.1:14450/refreshdata",{params: {"name": this.selectedTabularDataName}})
            .then((response) => {
              // 在这里处理从目标 URL 获取到的数据
              // 可以将数据保存到组件的 data 中，以供其他地方使用
              const data = response.data;
              // 假设您将获取的数据保存在组件的 dataList 变量中
              this.dataList = data;
              // 返回一个 Promise.resolve() 表示请求成功
              return Promise.resolve();
            })
            .catch((error) => {
              // 请求失败时的处理
              console.error("Failed to get data from target URL:", error);
              // 返回一个 Promise.reject() 表示请求失败
              return Promise.reject(error);
            });
        },
        onBeforeUpload: function(file) {
            // let self = this
            // console.log('file', file)
            // const isJSON = file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
            // const isLt2M = file.size / 1024 / 1024 < 2;
            // let fileNameArray = this.getExistedFileNameArray()
            // let notExisted = (fileNameArray.indexOf(file.name) === -1)
            // if (!isJSON) {
            //   this.$message.error('The uploaded file must be JSON format!');
            //   return
            // }
            // if (!isLt2M) {
            //   this.$message.error('The file size can not exceed 2MB!');
            //   return
            // }
            // if (!notExisted) {
            //   this.$message.error('The file name is existed!');   
            //   return 
            // }
            // var reader = new FileReader();
            // reader.readAsText(file, 'utf-8');
            // reader.onload = function(evt) {
            //    let fileString = evt.target.result // content
            //    console.log('fileString', fileString)
            // }
            // return (isJSON && isLt2M && notExisted);
        },
        getExistedFileNameArray: function() {
            // TODO
            let fileNameArray = []
            return fileNameArray
        },
        handlePreview: function(file) {
        },
        handleUploadSuccess: function(res, file) {
          console.log("hhhhhhhhh test!")
        },
        addDataCallback: function(resData) {
        },
        handleRemove: function() {
        },
        handleDelete: function(index, row) {
            this.tabularDataObjList.splice(index, 1)
            let dataObj = {
                username: row.username,
                filename: row.filename,
                depth: row.depth
            }
            this.selectedTabularDataName = null
            this.tempSelection = null
        },
        removeDataCallback: function(resData) {
            this.promptMessage(resData.type, resData.message)
        },
        setCurrent(fileName) {
            for (let i = 0; i < this.tabularDataObjList.length; i++) {
                let treeDataObj = this.tabularDataObjList[i]
                if (treeDataObj.filename === fileName) {
                    let row = this.tabularDataObjList[i]
                    this.$refs.treeDataTable.setCurrentRow(row);
                    
                }
            }
        },
        updateSelectedTabularDatasetName: function (selectedFileName) {
            sysDatasetObj.updateSelectedTabularDataset(selectedFileName)
            this.$bus.$emit("update-selected-dataset")
        },
        updateSelectedRLDatasetName: function (selectedFileName) {
            sysDatasetObj.updateSelectedRLDataset(selectedFileName)
            this.$bus.$emit("update-selected-dataset")
        },
        promptMessage: function(type, message) {
            this.$message({
              type: type,
              message: message
            })
        }
    }
  }
</script>
<style lang="less">
  #refresh-dialog {
    .el-dialog__body {
      padding: 5px 20px !important;
    }
    .el-table td, .el-table th {
        padding: 1px 0 !important;
    }
    .el-button--mini, .el-button--mini.is-round {
        padding: 3px 5px;
    }
    .el-upload {
        width: 100%;
        .el-upload-dragger {
            width: 100%;
            height: 100px;
            .el-upload__text {
                line-height: 100px;
                width: 100%;
            }
        }
    }
  }
</style>
<style scoped lang="less">
  #refresh-dialog {
    padding-bottom: 5px;
    .inner-label {
      margin-top: 10px;
      margin-bottom: 10px;
      font-size: 1.2rem;
    }
    .content-container {
      margin-bottom: 10px;
      text-align: left;
      position: relative;
      height: 230px;
      top: 0px;
    }
    .dialog-footer {
      text-align: right;
      .el-button {
        padding: 6px 10px;
        border-radius: 0px;
      }
    }
  }
</style> -->
