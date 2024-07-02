<template>
  <div id="app" v-if="!loadingData">
    <el-menu
      class="el-menu-demo"
      mode="horizontal"
      background-color="#e6e6e6"
      text-color="#000"
      active-text-color="#0087d0"
    >
      <el-menu-item id="title">
        {{ appName }}
      </el-menu-item>

      <!-- <el-menu-item>
        <el-upload
          class="upload-demo"
          action="http://localhost:8080/"
          :limit="1"
          show-file-list="false"
          :on-success="handleUploadSuccess"
          :on-preview="handlePreview"
          :on-remove="handleRemove"
          :before-upload="onBeforeUpload"

        >
          <img :src="'./icon/upload.svg'" class="icon" />
        Upload Your Data
        </el-upload>
      </el-menu-item> -->

      <el-menu-item
        v-for="(operation, index) in operationArray"
        :key="operation"
        @click="changeDialogVisible(operation)"
      >
        <img :src="iconPath[index]" class="icon" />
        {{ operation }}
      </el-menu-item>     
      
      <el-menu-item
        style="width: 15%; right: 50px !important"
        class="zoom-operator"
      >
        <el-slider
          v-model="zoomScale"
          :min="10"
          :max="150"
          :step="2"
          @input="handle_zoom_scale"
        ></el-slider>
      </el-menu-item>
      <el-menu-item class="zoom-operator" @click="handle_zoom()">
        <img :src="get_zoom_icon()" class="icon" />
      </el-menu-item>
    </el-menu>

    <!-- <div class="content-container">
      <TableView :isHeaderFixed="isHeaderFixed" @changeHeaderFixed="change_is_header_fixed($event)"></TableView> 
    </div> -->

    <div
      class="content-container"
      :class="{ 
        'content-container-right-margin': showVisPanel, 
        'content-container-bottom-margin': showRLPanel}"
    >
      <TableView :key="tableViewKey"></TableView>
    </div>

    <div
      id="vis-panel"

      :class="{
        'vis-panel-slide-in': showVisPanel,
        'vis-panel-slide-out': !showVisPanel,
      }"
    >
      <VisView></VisView>
    </div>

    <!-- <div
      id="rl-panel"
      :class="{'rl-panel-right-margin': showVisPanel}"
      v-if="showRLPanel">
      <RLPanel :insightList="insightList" :key="tableViewKey"></RLPanel>
    </div> -->
    <div
      id="rl-panel"
      :class="{'rl-panel-right-margin': showVisPanel}">
      <RLPanel :insightList="insightList" :key="tableViewKey"></RLPanel>
    </div>

    <el-dialog
      title="Example Dataset"
      id="dataset-dialog"
      :visible.sync="datasetDialogVisible"
    >
      <DataDialog :datasetDialogKey="datasetDialogKey"> </DataDialog>
    </el-dialog>

    <el-dialog
      title="Upload"
      id="upload-dialog"
      :visible.sync="uploadDialogVisible"
    >
      <UploadDialog> </UploadDialog>
    </el-dialog>

    <el-dialog :title="dialogTitle" :visible.sync="showDialog" width="30%">
      <span>{{ dialogText }}</span>
      <span slot="footer" class="dialog-footer">
        <el-button @click="CancelDialog">No</el-button>
        <el-button type="primary" @click="ConfirmDialog">Yes</el-button>
      </span>
    </el-dialog>

    <!-- <el-dialog
      title="Refresh Dataset"
      id="refresh-dialog"
      :visible.sync="refreshDialogVisible"
    >
      <RefreshDialog :refreshDialogKey="refreshDialogKey"> </RefreshDialog>
    </el-dialog> -->

    <!-- <button title="Refresh Dataset" id="refresh-dialog" @click="refreshFile" :visible.sync="refreshDialogVisible"></button> -->

    <el-dialog
      title="Select RL Example"
      id="selectRL-dialog"
      :visible.sync="SelectRLDialogVisible"
    >
      <SelectRLDialog > </SelectRLDialog>
      <!-- <SelectRLDialog :selectRLDialogKey="selectRLDialogKey"> </SelectRLDialog> -->
    </el-dialog>
    <!-- <button title="Select RL Example" id="selectRL-dialog" @click="selectRLData" :visible.sync="SelectRLDialogVisible"></button> -->
  
    <!-- <el-dialog :title="'User Guidance'" :visible.sync="showVideo" width="1080px">
      <div style="color:red">必填！！！</div>
      <a href="https://www.wjx.cn/vj/h4Ln1n6.aspx">用户问卷调查 https://www.wjx.cn/vj/h4Ln1n6.aspx</a>
      <br>
      <br>
      <video height="540px" controls>
        <source :src="videoPath" type="video/mp4">
      </video>
    </el-dialog> -->

    <el-dialog
      title="Export"
      id="export-dialog"
      :visible.sync="exportDialogVisible"
    >
      <ExportDialog></ExportDialog>
    </el-dialog>
  </div>
</template>

<script>
import TableView from "./views/TableView.vue";
import VisView from "./views/VisView.vue";
import { getTabularDataset, getRLDataset, geAlternativeDataset} from "@/communication/communicator.js";
import { parseTabularData } from "@/utils/tabularDataParser.js";
import { Dataset } from "@/dataset/dataset.js";
import DataDialog from "@/views/dialogs/DataDialog.vue";
import RefreshDialog from "@/views/dialogs/RefreshDialog.vue";
import SelectRLDialog from "@/views/dialogs/SelectRLDialog.vue";
import UploadDialog from "@/views/dialogs/UploadDialog.vue";
import ExportDialog from "@/views/dialogs/ExportDialog.vue";
import RLPanel from "./views/RLPanel.vue"
import { mapState, mapMutations } from "vuex";
import axios from 'axios';
import { contains } from "vega-lite";
// import { getPandasData } from './communication/communicator';

export default {
  name: "app",
  components: {
    VisView,
    TableView,
    DataDialog,
    SelectRLDialog,
    RefreshDialog,
    UploadDialog,
    ExportDialog,
    RLPanel
  },
  computed: {},
  data() {
    // console.error("null_insightList-new-return");
    return {
      appName: "InsighTable",
      // operationArray: ["Open Example Data", "Upload Your Data", "Export", "Open RL Example", "Select RL Example", "Refresh Result"],
      operationArray: ["Open Example Data", "Upload Your Data", "Export"],
      iconPath: [
        "./icon/open-file.svg",
        "./icon/upload.svg",
        "./icon/save.svg",
        // "./icon/RL.svg",
        // "./icon/RL.svg",
        // "./icon/datalist.svg",
        // "./icon/refresh.svg" 
      ],
      // activeIndex: "",
      datasetDialogVisible: false,
      SelectRLDialogVisible: false,
      refreshDialogVisible: false,
      uploadDialogVisible: false,
      exportDialogVisible: false,
      datasetDialogKey: 0,
      loadingData: true,
      loading_rl: true,
      rlDataDeferObj: null,

      initializeVis: false,
      showVisPanel: false,
      showRLPanel: false,

      showDialog: false,
      dialogTitle: "",
      dialogText: "",

      isZoomOut: false,

      zoomScale: 100,
      tableViewKey: 1,

      showVideo: true,
      videoPath:"http://r9ea0k3fo.hb-bkt.clouddn.com/guidance.mp4",

      insightList: [],
      rlobj_list: null,
      dataframe: null,
      alternative_rlobj_list: []
    };
  },
  beforeMount: function () {
    let self = this;
    window.sysDatasetObj = new Dataset();
    let tabularDataDeferObj = $.Deferred();
    $.when(tabularDataDeferObj).then(function () {
      self.loadingData = false;
    });
    let tabularDataList = ["*"];
    // initialize the tabular dataset
    // getTabularDataset(
    //   tabularDataList,
    //   function (datalist) {
    //     let dataobj_list = parseTabularData(datalist) 
    //     sysDatasetObj.updateTabularDatasetList(dataobj_list);
    //     // tabularDataDeferObj.resolve();
    //   }
    // ),
    
    getRLDataset(
      function (rl_res) {
        let rl_obj_list = parseTabularData(rl_res)
        sysDatasetObj.updateRLDatasetList(rl_obj_list)
        tabularDataDeferObj.resolve();
      }
    )

    // getDFList(
    //   function(df_list){
    //     sysDatasetObj.updateDFList(df_list)
    //   }
    // )

    geAlternativeDataset(
      'Console',
      function (rl_res) {
        let rl_obj_list = parseTabularData(rl_res)
        console.log("updateAlterRLDatasetList")
        sysDatasetObj.updateAlterRLDatasetList('Console', rl_obj_list)
      }
    )

    geAlternativeDataset(
      'Income',
      function (rl_res) {
        let rl_obj_list = parseTabularData(rl_res)
        console.log("updateAlterRLDatasetList")
        sysDatasetObj.updateAlterRLDatasetList('Income', rl_obj_list)
      }
    )

    this.selectedRLDataName = 'Console'

    // getRefreshedData(
    //   function (rl_refresh) {
    //     let rl_refresh_obj_list = parseTabularData(rl_refresh)
    //     sysDatasetObj.updateRLDatasetList(rl_refresh_obj_list)
    //     tabularDataDeferObj.resolve();
    //   }
    // )
    // this.get_insight_list()
    // this.showRLPanel = true
  },
  mounted: function () {
    let self = this;
    this.$bus.$on("visualize-selectedData", () => {
      this.initializeVis = true;
      this.showVisPanel = true;
    });

    this.$bus.$on("update-selected-dataset", () => {
      this.tableViewKey = (this.tableViewKey + 1) % 2;
      // console.log("update selected dataset", this.tableViewKey);
    });

    this.$bus.$on("select-canvas", () => {
      this.showVisPanel = true;
      // console.log("showVisPanel-select-canvas", this.showVisPanel);
    });

    this.$bus.$on("show-dialog", (data) => {
      this.showDialog = true;
      this.dialogTitle = data.title;
      this.dialogText = data.text;
    });

    this.$bus.$on("close-data-dialog", (isConfirm) => {
      this.datasetDialogVisible = false;
      if (isConfirm && this.datasetDialogKey == 1) {
        this.get_insight_list()
        this.showRLPanel = true
      }
      if (this.datasetDialogKey == 0) {
        this.showRLPanel = false
      }
    });

    this.$bus.$on("close-upload-dialog", () => {
      this.uploadDialogVisible = false;
    });

    this.$bus.$on("close-VisView", () => {
      this.showVisPanel = false;
      console.log("showVisPanel-close-VisView", this.showVisPanel);
    });
    this.$bus.$on("change-zoomScale", (value) => {
      this.zoomScale = value;
      console.log("change-zoomScale", value);
    });
    this.$bus.$on("close-refresh", () => {
      this.refreshDialogVisible = false;
    });
    this.$bus.$on("close-SelectRL", (data_name) => {
      this.SelectRLDialogVisible = false;
      console.log('close-SelectRL', data_name)
      this.selectedRLDataName = data_name
    });
    this.$bus.$on("SelectRL", (data_name) => {
      console.log('SelectRL', data_name)
      this.selectedRLDataName = data_name
    });
    this.$bus.$on("alternative-find", (hp_pos) => {
      // 两个数 且 开
      console.log("alternative-find hp_pos", hp_pos)
      // console.log("alternative-find rec_list", this.alternative_rlobj_list)
      if (this.selectedRLDataName == 'Console')
        var data = sysDatasetObj.AlterConsoleDataObjList
      else if(this.selectedRLDataName == 'Income')
        var data = sysDatasetObj.AlterIncomeDataObjList
      else
        console.error("not found in AlterDataObjList")
      let alternative_list = []
      var type_haved = {}
      type_haved['Unit Visualization'] = true;
      for(var txt of data){
        // console.log("encodingencodingencoding", txt.encoding)
        for(var insight of txt.encoding){
          let tmp_pos_row = []
          let tmp_pos_col = []
          // console.log("pos_rowpos_rowpos_row", insight.pos_row, insight.pos_col)
          // 一个数 或者 开
          if (typeof insight.pos_row === 'number')
            tmp_pos_row = [insight.pos_row, insight.pos_row];
          else
            tmp_pos_row = [insight.pos_row[0], insight.pos_row[1]-1];
          if (typeof insight.pos_col === 'number')
            tmp_pos_col = [insight.pos_col, insight.pos_col];
          else
            tmp_pos_col = [insight.pos_col[0], insight.pos_col[1]-1];
          // console.log("hp_poshp_poshp_pos", hp_pos, tmp_pos_row, tmp_pos_row);
          if (hp_pos == undefined)
            console.error("hp_posundefinedundefined");
          let tmp_hp_pos;
          if (Array.isArray(hp_pos)) {
            // console.log("变量是数组");
            tmp_hp_pos = hp_pos;
          } else {
            // console.log("变量是字典");
            tmp_hp_pos = [[hp_pos["top"], hp_pos["bottom"]], [hp_pos["left"], hp_pos["right"]]];
          }
          hp_pos = tmp_hp_pos;
          // var selectedArea = {"top":encoding.hp_pos[0][0], "bottom":encoding.hp_pos[0][1], "left":encoding.hp_pos[1][0], "right":encoding.hp_pos[1][1]}
          let judge = this.judge_merge(hp_pos[0][0], hp_pos[1][0], hp_pos[0][1], hp_pos[1][1], tmp_pos_row[0], tmp_pos_col[0], tmp_pos_row[1], tmp_pos_col[1]);
          // console.log("judge_merge before", judge, hp_pos[0][0], hp_pos[1][0], hp_pos[0][1], hp_pos[1][0], tmp_pos_row[0], tmp_pos_col[0], tmp_pos_row[1], tmp_pos_col[1]);
          if (judge === false && insight.rec_list !== null)
            for(var rec of insight.rec_list){
              if (rec.length !== 5)
                console.error("sothing in rec_list pos length")
              // else if (rec[4] === true){
              else{
                // if(this.judge_merge(hp_pos[0][0], hp_pos[1][0], hp_pos[0][1], hp_pos[1][0], rec[0], rec[2], rec[1], rec[3]))
                //   console.log("judge_merge", hp_pos[0][0], hp_pos[1][0], hp_pos[0][1], hp_pos[1][0], rec[0], rec[2], rec[1], rec[3])
                let judge = this.judge_merge(hp_pos[0][0], hp_pos[1][0], hp_pos[0][1], hp_pos[1][1], rec[0], rec[2], rec[1], rec[3]);
                if(judge === true)
                  break;
                // console.log("judge_merge after", judge, hp_pos[0][0], hp_pos[1][0], hp_pos[0][1], hp_pos[1][0], rec[0], rec[2], rec[1], rec[3])
              }
              // console.log("insight.rec_list", rec)
            }
          if(judge === true){
            insight.hp_pos = [tmp_pos_row, tmp_pos_col]
            if (!!!(insight.vis_type in type_haved)){
              alternative_list.push(insight)
              type_haved[insight.vis_type] = true;
            }
            // console.log("judge true", insight)
            // this.update_insight_list(insight)
          }
        }
      }
      // console.log("alternative_list", alternative_list)
      console.log("type_haved", type_haved);
      this.$bus.$emit("send-alternative_list", alternative_list);
    });
    this.$bus.$on("update_next_encoding", (data) => (
      this.update_next_encoding(data)
    ));
    this.$bus.$on("App-alter_update_vis-end", (new_rl_list)=>{
      this.insightList = new_rl_list;
      console.log("App-alter_update_vis-end", this.insightList);
    });
    this.$bus.$on("clear-insightList", (val, insight_type=null) => {
      console.log("clear-insightList", val, insight_type);
      if (val != null && val != undefined && val != 0){
        this.insightList = this.insightList.slice(0, val);
        // console.error("null_insightList-clear-insightList", this.insightList.length);
      }
      else{
        // console.error("null_insightList-clear-insightList");
        this.insightList = [];
      }

      if (insight_type != null){
        this.insightList.push(insight_type);
        // console.log("hhhhhhclear-insightListclear-insightList", val, insight_type);
      }
      // else
      //  console.error("eeeeeeeeeclear-insightListclear-insightList", val, insight_type);
    });
  },
  methods: {
    // ...mapMutations([
    //   "UPDATE_HP_TEST"
    // ]),
    iconClass(operation) {
      return "icon-" + operation;
    },
    judge_merge(x1, y1, x2, y2, x3, y3, x4, y4){
      if (Math.max(x1, x3) <= Math.min(x2, x4) && Math.max(y1, y3) <= Math.min(y2, y4))
        return true;
      return false;
    },
    // closeDataDialog() {
    //   this.datasetDialogVisible = false
    // },
    async changeDialogVisible(panel_name) {
      // console.log("panel_name", panel_name);
      if (panel_name === "Open Example Data") {
        this.datasetDialogKey = 0
        this.datasetDialogVisible = true;
      }
      if (panel_name === "Upload Your Data") {
        this.uploadDialogVisible = true;
        // let tmp_zoom = this.zoomScale;
        // this.$bus.$emit("change-zoom", 98);
      }
      if (panel_name === "Export") {
        this.exportDialogVisible = true;
      }
      if (panel_name === "Open RL Example") {
        this.datasetDialogKey = 1
        this.datasetDialogVisible = true;
      }
      if (panel_name === "Refresh Result") {
        // let tmp_zoom = this.zoomScale;
        // // this.refreshDialogVisible = true;
        // // this.UPDATE_HP_TEST(this.alternative_rlobj_list)
        // this.$bus.$emit("change-current-auto-vis-num", -1);
        // // await this.$bus.$emit("remove-canvas");
        // // await this.$bus.$emit("clear-AllTableView");
        // console.log("change-current-auto-vis-num-before", -1);
        
        // this.loading_rl = true
        // // self.rlDataDeferObj = $.Deferred();
        // // $.when(self.rlDataDeferObj).then(function () {
        // //   self.loading_rl = false;
        // // });
        // await this.refreshFile();
        // await sysDatasetObj.updateRefreshedRLDataset(this.rlobj_list);
        // // console.log("sysDatasetObjupdateSelectDF", sysDatasetObj);
        // await sysDatasetObj.updateSelectDF(this.dataframe);
        // await this.$bus.$emit("update-selected-dataset");
        // this.tableViewKey = (this.tableViewKey + 1) % 2;
        // // console.log("update selected dataset", this.tableViewKey);
        // // console.log("this.$bus.$emit(update-selected-dataset);", this.tableViewKey);
        // // self.tableViewKey = 1;
        // // this.$bus.$emit("update-selected-dataset");
        // // this.$bus.$emit('close-data-dialog', true);
        // await this.get_insight_list();
        // this.showRLPanel = true;

        // // this.$bus.$emit("clear-visdb")

        // // // this.datasetDialogVisible = true;
        // // console.log("change-current-auto-vis-num-after", this.insightList.length);
        // // await this.$bus.$emit("change-current-auto-vis-num", this.insightList.length-1);
        // this.$bus.$emit("change-zoom", tmp_zoom);
      }
      // if (panel_name === "Select RL Example") {
      //   this.SelectRLDialogVisible = true;
      // }
    },
    ConfirmDialog() {
      this.showDialog = false;
      console.log("showVisPanel-ConfirmDialog", this.showVisPanel, this.dialogText);
      this.$bus.$emit("confirm-dialog", this.dialogText);
      // this.$bus.$emit("confirm-dialog-hp", this.dialogText);
    },
    CancelDialog() {
      this.showDialog = false;
      this.$bus.$emit("cancel-dialog", this.dialogText);
    },
    update_next_encoding(data){
      // TODO here 
      // var selectedDF = sysDatasetObj.selectedDF
      var selectedDF = this.dataframe
      // console.log("this.dataframe", typeof this.dataframe, this.dataframe)
      axios.post('http://127.0.0.1:14450/update_next_encoding', { 'insight_list':data, 'selectedDF':selectedDF})
      .then(response => {
        console.log("update_next_encoding_post", response.data);
        if (response.data["encoding"] != null){
          this.$bus.$emit("TableView-visualize-next-encoding", response.data["encoding"]);
          this.$bus.$emit("RLPanel-visualize-next-encoding", response.data["encoding"]);
        }
        // else
        //   console.error("thisinsight_both_listpush-null");
      })
      .catch(error => {
        console.error(error);
      });
    },
    async refreshFile() {
      console.log("refreshFile");
      // await axios
      //   .get("http://127.0.0.1:14450/refreshdata", {
      //     params: {"name": this.selectedRLDataName}
      //   })
      //   .then((response) => {
          
      //   // this.$bus.$emit("close-VisView");
      //   // this.$bus.$emit("clear-selectedCell");
      //   //   let rl_obj_list = parseTabularData(rl_res)
      //   // sysDatasetObj.updateRLDatasetList(rl_obj_list)
      //   // tabularDataDeferObj.resolve();
      //     var datalist = response.data[0];
      //     var dataframe_json = response.data[1];
      //     console.log("datalist", datalist);
      //     console.log("dataframe_json", dataframe_json);
      //     datalist = JSON.parse(datalist);
      //     var dataframe = JSON.parse(dataframe_json);
      //     console.log("GET refreshdata dataframe", typeof dataframe, dataframe);
      //     // this.dataList = data;
      //     // 返回一个 Promise.resolve() 表示请求成功
      //     this.rlobj_list = parseTabularData(datalist) 
      //     this.dataframe = dataframe
      //     this.alternative_rlobj_list.concat(this.rlobj_list)
      //     // sysDatasetObj.updateTabularDatasetList(dataobj_list);
      //     // sysDatasetObj.updateSelectedRLDataset(dataobj_list)
      //     // console.log("dataobj_list", datalist);
      //     // sysDatasetObj.updateRefreshedRLDataset(dataobj_list);
      //     // this.datasetDialogKey = 1;
          
          
          
      //     // tabularDataDeferObj.resolve();
      //     return Promise.resolve();
      //   })
      //   .catch((error) => {
      //     // 请求失败时的处理
      //     console.error("Failed to get data from target URL:", error);
      //     // 返回一个 Promise.reject() 表示请求失败
      //     return Promise.reject(error);
      //   });
    },
    get_zoom_icon() {
      if (this.isZoomOut) {
        return "./icon/resume.svg";
      } else return "./icon/fitin.svg";
    },
    handle_zoom() {
      this.isZoomOut = !this.isZoomOut;
      this.$bus.$emit("change-zoom");
    },
    handle_zoom_scale(value) {
      this.$bus.$emit("change-zoom", value);
    },

    // get_insight_list() {
    //   var res = []
    //   var data = sysDatasetObj.selectedRLDataObj
    //   for (var item of data.encoding) {
    //     if(item.insight_type == 'Pearsonr') {
    //       item.insight_type = 'Pearson'
    //     }
    //     else if(item.insight_type == 'M-Top 2') {
    //       item.insight_type = 'Top 2'
    //     }
    //     else if(item.insight_type == 'M-Dominance') {
    //       item.insight_type = 'Dominance'
    //     }
    //     else if(item.insight_type == 'M-Evenness') {
    //       item.insight_type = 'Evenness'
    //     }
    //     res.push(item.insight_type)
    //   }
    //   this.insightList = res
    //   console.log('get_insight_list', this.insightList.length)
    //   // self.rlDataDeferObj.resolve()
    //   this.loading_rl = false
    // },
    // update_insight_list(insight) {
    //   var res = []
    //   var data = this.insightList
    //   console.log("update_insight_list", data)
    //   if(insight.insight_type == 'Pearsonr') {
    //     insight.insight_type = 'Pearson'
    //   }
    //   else if(insight.insight_type == 'M-Top 2') {
    //     insight.insight_type = 'Top 2'
    //   }
    //   else if(insight.insight_type == 'M-Dominance') {
    //     insight.insight_type = 'Dominan7ce'
    //   }
    //   else if(insight.insight_type == 'M-Evenness') {
    //     insight.insight_type = 'Evenness'
    //   }

    //   console.log("updateinsight", insight.hp_pos)
    //   console.log("updateinsight_rec", insight.rec_list)

    //   for (var item of data) {
    //     console.log("updateitem", item.pos_row, item.pos_col)
    //     console.log("updateitemrec_list", item.rec_list)
    //   }
    //   this.insightList = res
    //   console.log('get_insight_list', this.insightList.length)
    // },
  }, 
  watch: {
    // loading_rl: function (val) {
    //   let self = this
    //   if (!val) {
    //     this.tableViewKey = (this.tableViewKey + 1) % 2;
    //     console.log("update selected dataset", this.tableViewKey);
    //     console.log("this.$bus.$emit(update-selected-dataset);", this.tableViewKey);
    //     // this.datasetDialogVisible = true;
    //     self.$bus.$emit("clear-visdb")
    //     console.log("change-current-auto-vis-num-after", this.insightList.length);
    //     self.$bus.$emit("change-current-auto-vis-num", this.insightList.length-1);
    //   }
    // },
  },
  beforeDestroy(){
    this.$bus.$off("update_next_encoding");
    this.$bus.$off("clear-insightList");
    this.$bus.$off("visualize-selectedData");
    this.$bus.$off("App-alter_update_vis-end");
  }
};
</script>

<style lang="less">
@side-panel-width: 20%;
@bottom-panel-width: 10%;
@padding: 0.7rem;
@menu-height: 2.5rem;
@icon-size: 1.4rem;

html {
  font-size: 100%;
}
.zoom-operator {
  position: absolute !important;
  right: 10px !important;
}
#app {
  font-family: "Avenir", Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  position: absolute;
  top: 0%;
  bottom: 0%;
  left: 0%;
  right: 0%;
  overflow: hidden;
  .el-menu.el-menu--horizontal {
    border-bottom: none;
    .el-menu-item {
      height: @menu-height;
      line-height: @menu-height;
      font-size: 90%;
      padding: 0 10px;
      .icon {
        width: @icon-size;
        height: @icon-size;
      }
    }
    #title {
      font-weight: bolder;
      font-size: 115%;
    }
  }
  .el-slider__runway {
    background-color: white !important;
  }
  .vis-panel-slide-in {
    animation: slide-in 0.3s cubic-bezier(0.36, 0.07, 0.19, 0.97);
    transform: translateX(0);
  }
  .vis-panel-slide-out {
    animation: slide-out 0.3s cubic-bezier(0.36, 0.07, 0.19, 0.97);
    transform: translateX(100%);
    transition-delay: display 2s;
  }
  // .rl-panel-slide-in {
  //   animation: slide-in 0.3s cubic-bezier(0.36, 0.07, 0.19, 0.97);
  //   transform: translateX(0);
  // }
  // .rl-panel-slide-out {
  //   animation: slide-out 0.3s cubic-bezier(0.36, 0.07, 0.19, 0.97);
  //   transform: translateX(100%);
  //   transition-delay: display 2s;
  // }
  @keyframes slide-out {
    from {
      transform: translateX(0);
    }
    to {
      transform: translateX(100%);
    }
  }

  .content-container-right-margin {
    transition-delay: 0.3s;
    right: @side-panel-width !important;
  }
  .content-container-bottom-margin {
    transition-delay: 0.3s;
    bottom: @bottom-panel-width !important;
  }

  .rl-panel-right-margin {
    transition-delay: 0.3s;
    right: @side-panel-width !important;
  }

  .vis-panel-bottom-margin {
    transition-delay: 0.3s;
    bottom: @bottom-panel-width !important;
  }

  @keyframes slide-in {
    from {
      transform: translateX(100%);
    }
    to {
      transform: translateX(0);
    }
  }
  .content-container {
    position: absolute;
    top: @menu-height;
    left: 0;
    bottom: 0;
    right: 0;
    margin-right: @padding;
  }
  svg:not(:root) {
    overflow: visible;
  }
  #vis-panel {
    position: absolute;
    right: 0%;
    width: @side-panel-width;
    top: @menu-height;
    bottom: 0%;
    border-left: solid #efefef 1px;
    background-color: white;
  }

  #rl-panel {
    position: absolute;
    top: 90%; // 1 - @bottom-panel-width
    right: 0;
    bottom: 0%;
    left: 0%;
    height: @bottom-panel-width;
    border-top: solid #cecece 1px;
    // border-left: solid #cecece 1px;
    background-color: white;
  }

  // scroll bar
  ::-webkit-scrollbar {
    width: 5px;
    height: 5px;
    border-radius: 5px;
  }
  ::-webkit-scrollbar-thumb {
    background-color: #dcdfe6;
    border-radius: 5px;
    // visibility: hidden;
    &:hover {
      visibility: visible;
    }
  }
}
</style>
