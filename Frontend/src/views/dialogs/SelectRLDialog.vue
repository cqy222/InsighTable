<template>
    <div id = 'selectRL-dialog'>
        <div class = "content-container">
            <el-table 
                ref="treeDataTable"
                highlight-current-row
                :default-sort = "{prop: 'date', order: 'descending'}"
                @row-dblclick="dataTableRowDBClick"
                @row-click="dataTableRowClick"
                :data="RLDataNameList"
                max-height="200"
                border
                >
                <el-table-column
                  property="filename"
                  label="FileName"
                  sortable
                  fixed
                  >
                </el-table-column>
            </el-table>
        </div>
        <div slot="footer" class="dialog-footer">
          <el-button @click="closeSelectRLDialog">Cancel</el-button>
          <el-button type="primary" @click="confirmRLSelection">OK</el-button>
        </div>
    </div>
</template>

<script>
  import { mapState, mapMutations } from 'vuex'

  export default {
    name: 'SelectRLDialog',
    components: {},
    data() {
      return {
       search: ' ',
       fileList: [],
       selectedRLDataName: null,
       tempSelection: null,
       RLDataNameList: null,
      }
    },
    watch: {
    },
    props: ['datasetDialogKey'],
    created: function () {
    },
    beforeMount: function() {
      this.RLDataNameList = [{'filename': 'Console'}, {'filename': 'Income'}]
      console.log('RLDataNameList', this.RLDataNameList)
    },
    mounted: function() {
        // this.search = ""
        // this.selectedRLDataName = this.initTreeDataName
        // console.log('selectedRLDataName', this.selectedRLDataName)
        // this.setCurrent("Console Sales.xlsx")
    },
    computed: {
        ...mapState([
        ])
    },
    watch: {
      datasetDialogKey: function (new_val) {
        // if (new_val == 1) { // 将要显示的是RL结果弹窗 
        //   this.RLDataNameList = sysDatasetObj.RLDataObjList
        // }
        // else { // 将要显示的是普通表格文件弹窗
        //   this.RLDataNameList = sysDatasetObj.RLDataNameList
        // }
      },
    },
    methods: {
        dataTableRowDBClick: function() {
        },
        dataTableRowClick: function(row) {
            let fileName = row.filename
            this.tempSelection = fileName
            console.log('tempSelection', this.tempSelection)
            //  update the selected tabular dataset
            this.setCurrent(fileName)
        },
        closeSelectRLDialog: function() {
            this.$bus.$emit('close-SelectRL')
        },
        confirmRLSelection: function() {
            console.log('confirmRLSelection')
            // let self = this

            //  confirm the selected tabular dataset
            if (this.tempSelection != null) {
              // console.log("this.tempSelection", this.tempSelection)
              //   let selectionExisted = (this.treeDataArray.map(function(e) { return e.filename; })
              //     .indexOf(this.tempSelection) !== -1)
              //   console.log("selectionExisted", selectionExisted)
              //   if (selectionExisted) {
              //       this.selectedRLDataName = this.tempSelection
              //       this.updateSelectedTabularDatasetName(this.selectedRLDataName)
              //       this.tempSelection = null
              //       self.$cookies.set('selected-data-name', this.selectedRLDataName)
              //   }
              this.selectedRLDataName = this.tempSelection
              console.log('selectedRLDataName', this.selectedRLDataName)
              
              this.tempSelection = null
              // this.$cookies.set('selected-data-name', this.selectedRLDataName)
            }
            // this.myupdateSelectedTabularDatasetName(this.selectedRLDataName)
            this.$bus.$emit('close-SelectRL', this.selectedRLDataName)
        },
        getFile: function() {
            console.log('upload file ok')
        },
        // onBeforeUpload: function(file) {
        //     let self = this
        //     console.log('file', file)
        //     const isJSON = file.type === 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet';
        //     const isLt2M = file.size / 1024 / 1024 < 2;
        //     let fileNameArray = this.getExistedFileNameArray()
        //     let notExisted = (fileNameArray.indexOf(file.name) === -1)
        //     if (!isJSON) {
        //       this.$message.error('The uploaded file must be JSON format!');
        //       return
        //     }
        //     if (!isLt2M) {
        //       this.$message.error('The file size can not exceed 2MB!');
        //       return
        //     }
        //     if (!notExisted) {
        //       this.$message.error('The file name is existed!');   
        //       return 
        //     }
        //     var reader = new FileReader();
        //     reader.readAsText(file, 'utf-8');
        //     reader.onload = function(evt) {
        //        let fileString = evt.target.result // content
        //        console.log('fileString', fileString)
        //     }
        //     return (isJSON && isLt2M && notExisted);
        // },
        getExistedFileNameArray: function() {
            // TODO
            let fileNameArray = []
            return fileNameArray
        },
        handlePreview: function(file) {
        },
        handleUploadSuccess: function(res, file) {
        },
        addDataCallback: function(resData) {
        },
        handleRemove: function() {
        },
        handleDelete: function(index, row) {
            this.RLDataNameList.splice(index, 1)
            let dataObj = {
                username: row.username,
                filename: row.filename,
                depth: row.depth
            }
            this.selectedRLDataName = null
            this.tempSelection = null
        },
        removeDataCallback: function(resData) {
            this.promptMessage(resData.type, resData.message)
        },
        setCurrent(fileName) {
            console.log('setCurrent(fileName)', fileName)
            for (let i = 0; i < this.RLDataNameList.length; i++) {
                let treeDataObj = this.RLDataNameList[i]
                if (treeDataObj.filename === fileName) {
                    let row = this.RLDataNameList[i]
                    this.$refs.treeDataTable.setCurrentRow(row);
                }
            }
        },
        // updateSelectedTabularDatasetName: function (selectedFileName) {
        //     sysDatasetObj.updateSelectedTabularDataset(selectedFileName)
        //     this.$bus.$emit("update-selected-dataset")
        // },
        // updateSelectedRLDatasetName: function (selectedFileName) {
        //     sysDatasetObj.updateSelectedRLDataset(selectedFileName)
        //     this.$bus.$emit("update-selected-dataset")
        // },
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
  #selectRL-dialog {
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
  #selectRL-dialog {
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
</style>
