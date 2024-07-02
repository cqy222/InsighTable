<template>
<div>
  <el-tabs v-model="activeName" type="card" @tab-click="handleClick">
    <el-tab-pane label="Template" name="first"></el-tab-pane>
    <el-tab-pane label="Alternative Insight" name="second"></el-tab-pane>
  </el-tabs>
  <div id="visview-container" v-if="activeName === 'first'">
      <div class="vis-test">
        {{ this.vegaConfig }}
      </div>

      <div id="vis-view">
        <!-- return buttons -->
        <el-row
          type="flex"
          justify="space-around"
          style="margin-left: 1px; margin-bottom: 0px"
        >
          <el-col :span="2">
            <el-button
              v-if="showPanelView && !showTemplates"
              @click="ClickReturnButton"
              type="text"
              size="medium"
            >
              <i class="el-icon-back"></i>
            </el-button>
          </el-col>
          <el-col :span="20">
            <el-button
              @click="DisableTableUnit"
              size="mini"
              plain
              style="margin-top: 0px; color: #d81e05"
            >Disable Selected Cells <svg
                style="transform: translateY(3px); margin-top: -5px"
                t="1648206142678"
                class="icon"
                viewBox="0 0 1024 1024"
                version="1.1"
                xmlns="http://www.w3.org/2000/svg"
                p-id="3218"
                width="15"
                height="15"
              >
                <path
                  d="M64 66.133333a42.496 42.496 0 0 0 0 60.16l55.04 55.466667v593.066667c0 46.933333 38.4 85.333333 85.333333 85.333333h593.066667l97.706667 97.706667a42.496 42.496 0 1 0 60.16-60.16L124.16 66.133333c-8.106667-8.106667-18.773333-12.373333-29.866667-12.373333s-22.186667 4.266667-30.293333 12.373333z m225.706667 708.693334h-85.333334v-85.333334h85.333334v85.333334z m0-170.666667h-85.333334v-85.333333h85.333334v85.333333z m-85.333334-170.666667v-85.333333h85.333334v85.333333h-85.333334z m256 341.333334h-85.333333v-85.333334h85.333333v85.333334z m-85.333333-170.666667v-85.333333h85.333333v85.333333h-85.333333z m170.666667 170.666667v-85.333334h81.066666l85.333334 85.333334h-166.4z m-170.666667-597.333334h85.333333v85.333334h-19.2l104.533334 104.533333v-19.2h298.666666c23.466667 0 42.666667 19.2 42.666667 42.666667v317.866666l85.333333 85.333334V348.16c0-46.933333-38.4-85.333333-85.333333-85.333333h-341.333333v-85.333334c0-46.933333-38.4-85.333333-85.333334-85.333333H270.506667l104.533333 104.533333v-19.2z m341.333333 256h85.333334v85.333334h-85.333334v-85.333334z"
                  p-id="3219"
                  fill="#d81e06"
                ></path>
              </svg>
            </el-button>
          </el-col>
          <el-col :span="4">
            <el-button @click="CloseVisPanel" type="text" size="medium"
            >
              <i class="el-icon-close"></i>
            </el-button>
          </el-col>
        </el-row>

        <div v-if="showUnitPanel">
          <unit-view :visData_arr="unitData_arr" :figID="this.figID"
            :recommendValue="recommendValue"></unit-view>
        </div>
        <div v-else-if="showTemplates">
          <templates-view
            v-on:select-template="SelectTemplate"
            :templates="this.templates"
          ></templates-view>
        </div>
        <div v-else>
          <!-- 使用v-if而不是v-show，否则值会更新不上来 -->
          <div v-if="showPanelView">
            <div class="unit-chart">
              <svg id="chart"></svg>
            </div>
            <panel-view
              :selections="this.ECSelections"
              :vegaConfig="this.vegaConfig"
              :recommendValue="recommendValue"
              v-on:apply-config="PreviewVegaConf"
              v-on:apply-vis="ApplyVis2Table"
            ></panel-view>
          </div>
        </div>
      </div>

      <div id="vg-tooltip-element"></div>
      <div id="unit-tooltip-element">
        <table>
          <tr>
            <td class="key">value</td>
            <td class="value">10</td>
          </tr>
        </table>
      </div>
  </div>
  <div v-else-if="activeName === 'second'" class="view-container">
      <el-col :span="4" :offset="20">
            <el-button @click="CloseVisPanel" type="text" size="medium">
          <i class="el-icon-close"></i>
        </el-button>
      </el-col>
      <!-- <templates-view
          v-on:select-template="SelectAlternativeTemplate"
          :templates="this.templates"
      ></templates-view> -->
        <!-- Bar Chart -->
      <!-- @changeData="changeData" -->
      <!-- <div :alternative_list="this.alternative_list" v-for="item in alternative_list" :key="item.index" class="alter-unit-chart">
        <svg :id="`alter-chart-${item.index}`"></svg>
      </div> -->
      <!-- <svg :id="`alter-chart-${item.index}`"></svg> -->
      <!-- <div :alternative_list="this.alternative_list" :key="this.alterKey"> -->
        <!-- this.alternative_unitflag_list -->
        <!-- <unit-view :visData_arr="unitData_arr" :figID="this.figID"
            :recommendValue="recommendValue"></unit-view> -->
      <div :key="alterKey">
        <div :key="index" v-for="(item, index) in alternative_list"  @click="alter_update_vis(index)" class="alter-unit-chart">
          <!-- <unit-chart v-if="alternative_unitflag_list[index]"></unit-chart> -->
          <!-- <svg  viewBox="0 0 1000 1000"> -->
          
          <svg v-if="alternative_unitflag_list[index]" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <!-- <rect width="100%" height="100%" fill="#f0f0f0" /> -->
            <!-- <line x1="0" y1="0" x2="200" y2="200" style="stroke: #dddddd; stroke-width: 4" /> -->

            <line x1="50" y1="0" x2="50" y2="200" style="stroke: #dddddd; stroke-width: 4" />
            <line x1="150" y1="0" x2="150" y2="200" style="stroke: #dddddd; stroke-width: 4" />
            <line x1="0" y1="50" x2="200" y2="50" style="stroke: #dddddd; stroke-width: 4" />
            <line x1="0" y1="150" x2="200" y2="150" style="stroke: #dddddd; stroke-width: 4" />
            <circle cx="100" cy="100" r="30" fill="#3498db" />
          </svg>
          
          <svg v-else :id="alter_chart_(index)" class="hp-test-chart"></svg>
        </div>
      </div>
      
      <!-- <div class="alter-unit-chart">
        {{ alternative_list }}
      </div> -->
          <!-- <svg id="alter-chart-0"></svg> -->
      <!-- <div v-if="showUnitPanel">
          <unit-view :visData_arr="unitData_arr" 
            :recommendValue="recommendValue"></unit-view>
        </div> -->
  </div>
</div>
</template>

<script>
import vegaEmbed from "vega-embed";
import PanelView from "./vis/PanelView.vue";
import TemplatesView from "./vis/TemplatesView.vue";
import UnitView from "./vis/UnitView.vue";
import {
  GetTemplates,
  GetTemplate,
  VegaTemplate,
  supportedTemplate,
  FieldSelection
} from "./vis/TemplateCompiler";
// import {
//   hp_get_vega_vis,
//   hp_get_unit_vis
// } from "./TableView.vue"
import { EncodingCompiler } from "./vis/SchemaCompiler";
import { VisDatabase } from "./vis/VisDatabase";
import { DialogTexts } from "./dialogs/DialogTexts";
import { UnitCompiler, getColorFunction } from "./vis/UnitCompiler"
import { mapState } from "vuex";
import { rangeType } from 'vega-lite/build/src/channel';
import { field, stringValue } from "vega";
// visualize-selectedData -> visView -> TemplateView ->(vegaConfig) visView -> Panel -> (metaData+vegaConfig+data) VisDataBase -> visualization
// visualize-data -> visView (template) -> VisTemplates -> (vegaConfig) visView -> Panel -> (meataData+vegaConfig+data) VisDataBase -> visualization
export default {
  name: "VisView",
  components: {
    PanelView,
    TemplatesView,
    UnitView,
    stringValue
},
  computed: {
    ...mapState([
      "hp_test"
    ]),
    VegaConfigNoData() {
      return { mark: this.vegaConfig.mark, encoding: this.vegaConfig.encoding };
    },
    vegaConfig() {
      return this.currentTemplate.GetVegaConfig();
    },
    ECSelections() {
      return this.currentTemplate.GetSelections();
    },
  },
  watch: {
    hp_test: function() {
      console.log("render in the visview.vue")
    }
  },
  data() {
    return {
      showTemplates: false,
      showUnitPanel: false,
      showPanelView: false,

      visData: {}, // data from visualize selected data
      metaData: {},
      position: {},
      mytest:{
        $schema: "https://vega.github.io/schema/vega-lite/v5.json",
        data:{
          values: [
            {a: "A", b:28},
            {a: "B", b:28},
            {a: "C", b:28},
            {a: "D", b:28},
            {a: "E", b:28},
            {a: "F", b:28},
            {a: "G", b:28},
            {a: "H", b:28},
            {a: "I", b:28},
          ]
        },
        mark: "bar",
        encoding:{
          x: {field: "a", type: "ordinal"},
          y: {field: "b", type: "quantitative"},
        }
      },

      // 1. selectTemplate -> currentTemplate.GetVegaConfig -> PanelView (tweakedData) -> VisView -> currentTemplate.CompileTweakedConfig (vega-lite) -> visualize -> visDB
      // 2. recommand -> templateName+recommandArea(meataData+visData) -> template -> visualize -> visDB
      currentTemplate: new VegaTemplate(),
      templates: [],

      VisDB: new VisDatabase(this.$bus),
      figID: "",
      lastFigID: "",

      currentGroupID: "",

      unitData_arr: [],
      recommendData_arr: [],

      recommendValue: {priority: [0, 2], type: "name", direction:"column"},

      activeName: 'first',
      alternative_list: [],
      alternative_unitflag_list: [],
      // alternative_list : [
      //     { id: 1, name: 'file1.txt', content: 'Content of file 1' },
      //     { id: 2, name: 'file2.txt', content: 'Content of file 2' },
      //     // ...
      //   ]
      alterKey: null,



      shapes: [
        { value: "circle", label: "circle" },
        { value: "square", label: "square" },
        { value: "triangle", label: "triangle" },
      ],
      scales: [
        { value: "linear", label: "linear" },
        { value: "pow", label: "pow" },
        { value: "log", label: "log" },
        { value: "sqrt", label: "sqrt" },
      ],
      scale: "linear",
      shape: "circle",
      color: "#00B2FF",
      relativeSize: 0.8,
      disabledEncodings: [
        { name: "color" },
        { name: "height" },
        { name: "width" },
        { name: "xOffset" },
        { name: "yOffset" },
      ],
      enabledEncodings: [{ name: "size" }, { name: "color" }],
      align: "middle",
    };
  },
  methods: {
    alter_chart_(i){
      return "alter_chart_" + i;
    },
    handleClick(tab) {
      // 处理选项卡点击事件
      this.activeName = tab.name;
      if(this.activeName === 'second'){
        this.alterKey = (this.alterKey + 1) % 2;
        this.$bus.$emit("alter-preview-config");
      }
      else if(this.activeName == 'first'){
        this.$bus.$emit("preview-config");
      }
    },
    update_activeName_first() {
      this.activeName = 'first';
      // console.log("update_activeName_first");
      // this.alterKey = (this.alterKey + 1) % 2;
      // this.$bus.$emit("alter-preview-config");
    },
    update_activeName_second() {
      this.activeName = 'second';
      // console.log("update_activeName_second");
      // this.alterKey = (this.alterKey + 1) % 2;
      // this.$bus.$emit("alter-preview-config");
    },
    alter_update_vis(index){
      // console.log("alter_update_vis-send", index, this.alternative_list[index]);
      // 这里需要考虑是否删掉被修改的这个index之后的insightList
      // this.$bus.$emit("alter_update_vis_on", this.alternative_list[index]);
      console.log("VisView-alter_update_vis", this.alternative_list, index, this.alternative_list[index]);
      this.$bus.$emit("RLPanel-alter_update_vis_on", this.alternative_list[index]);
    },
    // GetConfig() {
    //   let encodings = {};
    //   this.disabledEncodings.forEach((element) => {
    //     encodings[element.name] = false;
    //   });
    //   this.enabledEncodings.forEach((element) => {
    //     encodings[element.name] = true;
    //   });
    //   return {
    //     encodings: encodings,
    //     shape: this.shape,
    //     color: this.color,
    //     relativeSize: this.relativeSize,
    //     scale: this.scale,
    //     align: this.align,
    //   };
    // },
    // PreviewUnitConfig() {
    //   let chart = document.getElementById("alter-preview-svg");
    //   let encodings = {};
    //   this.disabledEncodings.forEach((element) => {
    //     encodings[element.name] = false;
    //   });
    //   this.enabledEncodings.forEach((element) => {
    //     encodings[element.name] = true;
    //   });

    //   let config = this.GetConfig();
    //   let height = 200;
    //   let width = 350;

    //   config.size = height * 0.5;
    //   config.height = height * 0.5;
    //   config.width = height * 0.5;

    //   config.xOffset = 0;
    //   config.yOffset = 0;
    //   config.frameHeight = 400;
    //   config.frameWidth = 700;

    //   if (config.encodings.size) {
    //     config.size = height * this.relativeSize;
    //     config.height = height * this.relativeSize;
    //     config.width = height * this.relativeSize;
    //   }
    //   if (config.encodings.height) {
    //     config.height = height * this.relativeSize;
    //   }
    //   if (config.encodings.width) {
    //     config.width = width * this.relativeSize;
    //   }

    //   if (config.encodings.xOffset) {
    //     config.xOffset = 100;
    //   }
    //   if (config.encodings.yOffset) {
    //     config.yOffset = 100;
    //   }

    //   if (config.encodings.opacity) {
    //     config.opacity = 0.8;
    //   }

    //   // if (config.encodings.color) {
    //   //   this.ShowMappedColor();
    //   // } else {
    //     this.showColorLegend = false;
    //   // }

    //   let preview = UnitCompiler.GetUnitDom(config);

    //   preview.setAttribute("transform", "translate(350,200)");

    //   // if (chart.childElementCount == 0) {
    //     chart.appendChild(preview);
    //   // } else {
    //   //   chart.childNodes[chart.childElementCount - 1].replaceWith(preview);
    //   // }
    // },

    CloseVisPanel() {
      console.log("close-VisView-CloseVisPanel");
      this.$bus.$emit("close-VisView");
    },
    // User Operation events

    OpenUnitView() {
      this.showUnitPanel = true;
      this.showTemplates = false;
      this.showPanelView = false;
    },

    // Input (data) and metadata to VisTemplates. Then get the templates. Open the template view.
    OpenTemplateView() {
      // console.log("GetTemplatesmetaData", this.metaData)
      // console.log("GetTemplatesvisData", this.visData)
      this.templates = GetTemplates(this.metaData, this.visData);
      // console.log("OpenTemplateView", this.templates)

      this.showTemplates = true;
      this.showUnitPanel = false;
      this.showPanelView = false;
    },

    // User select template from templateView, then update the vegaConfig
    SelectTemplate(template) {
      // console.log("select-template-start", template)
      this.currentTemplate = template;
      this.OpenPanelView();
    },

    SelectAlternativeTemplate(template) {
      // console.log("select-template-start", template)
      this.currentAlternativeTemplate = template;
      // this.OpenAlternativePanelView();
    },

    // Open panel view to tweak data
    OpenPanelView() {
      this.showTemplates = false;
      this.showUnitPanel = false;
      this.showPanelView = true;
      // console.log("preview_config-OpenPanelView");
      this.$bus.$emit("preview-config");
    },

    ClickReturnButton() {
      // 再写一个恢复，让点击return button后templateView自动读取metaData

      if (this.showTemplates) {
        return;
      }
      this.OpenTemplateView();
    },

    // User modified panel to update preview figure on top of the panel
    PreviewVegaConf(vegaConfig) {
      this.currentTemplate.CompileTweakedConfig(vegaConfig); // 可能有拷贝的问题
      // console.log("preview_config-PreviewVegaConf");
      this.$bus.$emit("preview-config"); // preview picture
    },

    DisableTableUnit() {
      this.VisDB.DisableTableUnit(
        this.position.height,
        this.position.width,
        this.position.x,
        this.position.y
      );
    },

    // Initially apply vega-lite config to the table, then register the config in database
    ApplyVis2Table(is_user = true) {
      // There is no generated data. So generate a new one.
      console.log("ApplyVis2Table", this.position, "=====", this.hp_pos)
      if (is_user === true){
        console.log("RLPanel-add-user-icon-send", this.hp_pos, this.position, this.currentTemplate, this.visData, this.metaData)
        this.$bus.$emit("RLPanel-add-user-icon", this.currentTemplate.name);
      }



      if (this.figID == "") {
        this.figID = this.VisDB.GenFig(
          this.position.height,
          this.position.width,
          this.position.x,
          this.position.y,
          this.currentTemplate,
          this.visData,
          this.metaData,
          this.hp_pos
        );
        // console.log("this.currentTemplate", this.currentTemplate)
        // console.log("this.position.height, this.position.width", this.position.height, this.position.width, this.position.x, this.position.y, this.currentTemplate, this.visData, this.metaData)
        this.$bus.$emit("auto-gen-new-fig-id", this.figID);  // 用于记录当前创建的figid方便在tableview中获取id
      } else {
        this.VisDB.SetTemplate(this.figID, this.currentTemplate);
        // console.log("RerenderCanvas-ApplyVis2Table");
        this.VisDB.RerenderCanvas(this.figID);
        if (this.VisDB.GetGroupMembers(this.figID).length > 1) {
          this.$bus.$emit("show-dialog", {
            title: "Batch Operation",
            text: DialogTexts.reconf,
          });
        }
      }
      
      this.$bus.$emit("apply-config", this.currentTemplate.name);
      // user-add-to-figlist
      // console.log("this.currentTemplate", this.currentTemplate);
    },

    ApplyUnitVis2Table(unitConfig) {
      this.visData = UnitCompiler.GetUnits(this.unitData_arr, unitConfig);
      // Use parents' figID to detect whether we should generate new data, and preserver current ID by using variable local
      console.log("ApplyUnitVis2Table", this.figID);
      // this.$bus.$emit("auto-gen-new-fig-id", ID);
      if (!!this.figID && this.figID !== "") {
        for (let i = 0; i < this.visData.length; i++) {
          let db = this.VisDB.database[this.visData[i].id];
          this.VisDB.RerenderCanvas(
            this.visData[i].id,
            db.x,
            db.y,
            db.height,
            db.width,
            this.visData[i].dom
          );
        }
      } else {
        let groupId;
        let ID;

        if (unitConfig.isrec) { // 是推荐的unit vis
          groupId = this.lastFigID
        }
        for (let i = 0; i < this.visData.length; i++) {
          let position = this.visData[i].position;
          let dom = this.visData[i].dom;
          ID = this.VisDB.GenUnit(
            position.height,
            position.width,
            position.x,
            position.y,
            dom,
            this.visData[i].value,
            this.hp_pos
          );
          if (ID) {
            this.visData[i].id = ID;
            // console.error("AddGroupMember-ApplyUnitVis2Table", groupId, ID);
            groupId = this.VisDB.AddGroupMember(groupId, ID);
            if (this.lastFigID != groupId) {
              this.$bus.$emit("auto-gen-new-fig-id", ID);  // 用于记录当前创建的figid方便在tableview中获取id
            }
            this.lastFigID = groupId;
          }
        }
        this.figID = ID;
        // this.$bus.$emit("auto-gen-new-fig-id", groupId)  // 用于记录当前创建的figid方便在tableview中获取id
      }
    },

    changeData(){

    }
  },
  mounted() {
    this.$bus.$on("clear-visdb", ()=> {
      this.VisDB = new VisDatabase(this.$bus)
    });

    this.$bus.$on("VisView-change-figID", (ID)=>{
      this.figID = ID;
    });

    this.$bus.$on("transmit-recommend-value-to-panel", (priority, type, direction) => {
      this.recommendValue.priority = priority
      this.recommendValue.type = type
      this.recommendValue.direction= direction
    });
    this.$bus.$on("send-alternative_list", (alternative_list) => {
      console.log("get-alternative_list", alternative_list.length, alternative_list);
      this.alternative_list = alternative_list;
      this.alternative_unitflag_list = [];
      for(var insight of this.alternative_list){
        // console.log("alternative_unitflag_list");
        if(insight.vis_type == "unit visualization")
          this.alternative_unitflag_list.push(true);
        else
          this.alternative_unitflag_list.push(false);
      }
      
      // console.log("this.alternative_unitflag_list", this.alternative_unitflag_list.length, this.alternative_list.length)
      this.alterKey = (this.alterKey + 1) % 2;
      // this.PreviewUnitConfig();

      // console.log("this.alterKey", this.alterKey);
      this.$bus.$emit("alter-preview-config");
      this.changeData();
    });

    this.$bus.$on("select-cell", () => this.VisDB.CancelAllSelections());
    this.$bus.$on("change-header", () => this.VisDB.RemoveAllCanvas());
    this.$bus.$on("remove-canvas", () => this.VisDB.RemoveAllCanvas());
    
    this.$bus.$on("delete-rl-fig-id", (id) => {   // 删除rl list不应存在的可视化
      let groupid = this.VisDB.GetGroupID(id);
      console.log("delete - id", id, groupid);
      if (groupid != null)  {
        this.VisDB.DeleteGroup(groupid);
      }
      else {
        this.VisDB.RemoveCanvas(id);
      }
    });
    this.$bus.$on("hidden-rl-fig-id", (id) => {
      // console.log("hidden - id", id)
      let groupid = this.VisDB.GetGroupID(id)
      if (groupid != null)  {
        this.VisDB.HiddenGroup(groupid);
      }
      else {
        this.VisDB.HiddenCanvas(id);
      }
    });
    this.$bus.$on("display-rl-fig-id", (id) => {
      // console.log("display - id", id)
      let groupid = this.VisDB.GetGroupID(id)
      if (groupid != null)  {
        this.VisDB.DisplayGroup(groupid);
      }
      else {
        this.VisDB.DisplayCanvas(id);
      }
    });
    

    // this.$bus.$on("delete-lick-close-button", (id) => {
    //   if (this.VisDB.GetGroupMembers(id).length > 1) {
    //     this.$bus.$emit("remove-groupCanvas", this.VisDB.GetGroupID(id));
    //     this.VisDB.RemoveGroupMember(id);
    //   }
    //   this.VisDB.RemoveCanvas(id);
    // });
    // console.log("confirm-dialog-on");
    this.$bus.$on("confirm-dialog", (dialogText) => {  // 确认对于推荐区域的批处理操作
      // console.log("confirm-dialog-remove", dialogText);
      if (dialogText == DialogTexts.remove) {
        this.VisDB.DeleteGroup(this.currentGroupID);
      } else if (dialogText == DialogTexts.reconf) {
        this.VisDB.ModifyGroupFigs(this.figID, this.currentTemplate);
      } else if (dialogText == DialogTexts.recommend) {
        // console.error("GenRecommendFigs-confirm-dialog", this.recommendData_arr);
        this.VisDB.GenRecommendFigs(
          this.recommendData_arr,
          this.currentTemplate,
          this.figID
        );
      }
    });
    this.$bus.$on("confirm-dialog-hp", (dialogText) => {  // 为了解决：可能是重名覆盖导致的问题
      // console.log("confirm-dialog-remove", dialogText);
      if (dialogText == DialogTexts.remove) {
        // console.log("remove-ins-marker", this.figID);
        // for (tmp_id of this.VisDB.GetGroupMembers(this.figID))
        //   this.$bus.$emit("remove-ins-marker", tmp_id);
        this.VisDB.DeleteGroup(this.currentGroupID);
      } else if (dialogText == DialogTexts.reconf) {
        this.VisDB.ModifyGroupFigs(this.figID, this.currentTemplate);
      } else if (dialogText == DialogTexts.recommend) {
        // console.error("GenRecommendFigs-confirm-dialog-hp", this.recommendData_arr);
        this.VisDB.GenRecommendFigs(
          this.recommendData_arr,
          this.currentTemplate,
          this.figID
        );
      }
    });
    this.$bus.$on("cancel-dialog", (dialogText) => {
      if (dialogText == DialogTexts.recommend) {
        this.$bus.$emit("clear-selectedCell");
        this.recommendData_arr = [];
      }
    });

    // Render figure on top of the side panel
    this.$bus.$on("preview-config", () => {
      if (this.showPanelView && this.activeName == 'first') {
      // if (this.showPanelView) {
        let height = document.getElementById("vis-panel").clientHeight * 0.25;
        let width = document.body.clientWidth * 0.19;
        // console.log("&&&&&&&&&&&&&&&&&&", height, width);
        let data = JSON.parse(
          JSON.stringify(this.currentTemplate.GetVegaLite(height, width, true))
        );
        // console.log("vegachart", data);
        // if (this.currentTemplate.name === supportedTemplate.NQ_Box_Plot){
        //   console.log("content.before", width, height);
        // }
        // lrf's version
        // vegaEmbed("#chart", data, {
        //   renderer: "svg",
        //   actions: false,
        // }).then(() => {
        //   let content = document.getElementById("chart").childNodes[0];
        //   let width = document.getElementById("chart").clientWidth;
        //   let height = document.getElementById("chart").clientHeight;
        //   console.log("height", height, "width", width);
        //   if (this.currentTemplate.name === supportedTemplate.NQ_Box_Plot){
        //     console.log("content.chart", width, height);
        //     console.log("content.contend", content, content.getBBox().width, content.getBBox().height);
        //   }
        //   if (this.currentTemplate.name === supportedTemplate.Q2_Horizon_Graph) {

        //   } else if (content.getBBox().width > width 
        //    || content.getBBox().height > height) {
        //     content.setAttribute(
        //       "transform",
        //       "translate(" +
        //         -5 +
        //         "," +
        //         -5 +
        //         ") scale(" +
        //         width / content.getBBox().width +
        //         "," +
        //         height / content.getBBox().height +
        //         ")"
        //     );
        //   }
        // });
        vegaEmbed("#chart", data, {
          renderer: "svg",
          actions: false,
        }).then(() => {
          console.log("chartchartchartchartchart", data);
          // let content = document.getElementById("chart").childNodes[0];
          // let width = document.getElementById("chart").clientWidth;
          // let height = document.getElementById("chart").clientHeight;
          // console.log("vegaEmbed encoding", data);
          // console.log("content.chart", width, height);
          // if (this.currentTemplate.name === supportedTemplate.NQ_Box_Plot){
          //   // console.log("content.chart", width, height);
          //   // console.log("content.contend", content, content.getBBox().width, content.getBBox().height);
          // }
          // if (this.currentTemplate.name === supportedTemplate.Q2_Horizon_Graph) {

          // } else if (content.getBBox().width > width 
          //  || content.getBBox().height > height) {
          //   console.log("vegaEmbed scale", content.getBBox().width, width, width / content.getBBox().width);
          //   console.log("vegaEmbed scale", content.getBBox().height, height, height / content.getBBox().height);
          //   // content.setAttribute(
          //   //   "transform",
          //   //   // "translate(" +
          //   //   //   -5 +
          //   //   //   "," +
          //   //   //   -5 +
          //   //   //   ") " + 
          //   //     "scale(" +
          //   //     (width + 0) / (content.getBBox().width ) / 1.2 +
          //   //     "," +
          //   //     (height + 0) / (content.getBBox().height ) / 1.1 +
          //   //     ")"
          //   // );
          // }
          // else{
          //   console.log("vegaEmbed else", this.currentTemplate.name);
          //   console.log("vegaEmbed getBBox()", content.getBBox(), width, height);
          // }
        });
      }
    });
    // this.$bus.$on("hp_get_vega_vis-return", (res)=>{
    //   // id, svgId, visdata, metadata, vis_type, direction = res
    //   // console.log("hp_get_vega_vis-returnvisdata", visdata);
    //   // console.log("hp_get_vega_vis-returnmetadata", metadata);
    //   this.$bus.$emit("embed-id-chat", res)
    // });
    
    // this.$bus.$on("hp_get_unit_vis-return", (res)=>{
    //   // id, svgId, visdata, metadata, vis_type, direction = res
    //   // console.log("hp_get_unit_vis-returnvisdata", visdata);
    //   // console.log("hp_get_unit_vis-returnmetadata", metadata);
    //   this.$bus.$emit("embed-id-chat", res)
    // });



    // this.$bus.$on("embed-id-chat", (id, svgId, visdata, metadata, vis_type, direction)=>{
    //   // console.log("embed-id-chat", id, svgId, visdata, metadata, vis_type, direction);
    //   console.log("embed-id-chat", svgId);
    //   visdata = JSON.parse(visdata);
    //   metadata = JSON.parse(metadata);

    //   // if (typeof metadata != Object) {
    //   //   metadata = JSON.parse(metadata);
    //   // }
    //   // console.log("GetTemplate-svgId", svgId)
    //   // console.log("GetTemplate-vis_type", vis_type)
    //   // console.log("GetTemplate-metadata", metadata)
    //   // console.log("GetTemplate-visdata", visdata)
    //   // console.log("GetTemplate-direction", direction)
    //   let currentTemplate = GetTemplate(vis_type, metadata, visdata, direction)
    //   console.log("currentTemplate", currentTemplate)
    //   // console.log("currentTemplate_name", vis_type)

    //   let height = document.getElementById("vis-panel").clientHeight * 0.25;
    //   let width = document.body.clientWidth * 0.19;
    //   let data = JSON.parse(
    //     JSON.stringify(currentTemplate.GetVegaLite(height, width))
    //   );
    //   // console.log("alter-preview-config data", data);
    //   // console.log("alter-preview-config height", height);
    //   // console.log("alter-preview-config width", width);
    //   // console.log("alter-preview-config", data, height, width);
    //   if(vis_type == "Unit Visualization"){
    //     // console.log("Unit_data", data.data)
    //     // console.log("Unit_encoding color", data.encoding.color)
    //     // console.log("Unit_encoding x", data.encoding.x)
    //     // console.log("Unit_encoding y", data.encoding.y)
    //   }
    //   else if(currentTemplate.name == supportedTemplate.NQor2Q_Simple_Line_Chart){
    //     data.encoding.x.field = "row 2";
    //     console.log("Line_Chart_data", data.data)
    //     console.log("Line_Chart_color", data.color)
    //     console.log("Line_Chart_encoding color", data.encoding.color)
    //     console.log("Line_Chart_encoding x", data.encoding.x)
    //     console.log("Line_Chart_encoding y", data.encoding.y)
    //     vegaEmbed(`#${svgId}`, data, {
    //       renderer: "svg",
    //       actions: false,
    //     }).then(() => {
    //       let content = document.getElementById(svgId).childNodes[0];
    //       let width = document.getElementById(svgId).clientWidth;
    //       let height = document.getElementById(svgId).clientHeight;
    //       if (
    //         content.getBBox().width > width ||
    //         content.getBBox().height > height
    //       ) {
    //         console.log("NQor2Q_Simple_Line_Chart.getBBox().width", content.getBBox().width, width);
    //         console.log("NQor2Q_Simple_Line_Chart.getBBox().height", content.getBBox().height, height);
    //         content.setAttribute(
    //           "transform",
    //           "translate(" +
    //             -5 +
    //             "," +
    //             -5 +
    //             ") "
    //             +
    //             "scale(" +
    //             1 +
    //             "," +
    //             height / content.getBBox().height +
    //             ")"
    //           );
    //         }
    //         else{
    //           console.log("vegaEmbed else")
    //         }
    //       });
    //   }
    //   else if(currentTemplate.name === supportedTemplate.NQ_Box_Plot){
    //     console.log("NQ_Box_Plot_direction", direction)
    //     // if(direction == "horizon")
    //     //   data.encoding.x = [];
    //     // else
    //     //   data.encoding.y = [];
    //     data.encoding.color = [];
    //     data.encoding.x = [];
    //     console.log("render Box_Plot", data);
    //     // console.log("NQ_Box_Plot_data", data.data)
    //     // console.log("NQ_Box_Plot_color", data.color)
    //     // console.log("NQ_Box_Plot_encoding color", data.encoding.color)
    //     // console.log("NQ_Box_Plot_encoding x", data.encoding.x)
    //     // console.log("NQ_Box_Plot_encoding y", data.encoding.y)

    //     vegaEmbed(`#${svgId}`, data, {
    //       renderer: "svg",
    //       actions: false,
    //     }).then(() => {
    //       let tmp = document.getElementById(svgId);
    //       console.log("tmpgetElementById", tmp);
    //       let content = document.getElementById(svgId).childNodes[0];
    //       let width = document.getElementById(svgId).clientWidth;
    //       let height = document.getElementById(svgId).clientHeight;
    //       if (
    //         content.getBBox().width > width ||
    //         content.getBBox().height > height
    //       ) {
    //         console.log("NQ_Box_Plot_direction.getBBox().width", content.getBBox().width, width);
    //         console.log("NQ_Box_Plot_direction.getBBox().height", content.getBBox().height, height);
    //         content.setAttribute(
    //           "transform",
    //           "translate(" +
    //             -5 +
    //             "," +
    //             -5 +
    //             ") " + 
    //             "scale(" +
    //             width / content.getBBox().width +
    //             "," +
    //             height / content.getBBox().height +
    //             ")"
    //           );
    //         }
    //         else{
    //           console.log("vegaEmbed else")
    //         }
    //     });
    //   }
    //   else if (currentTemplate.name === supportedTemplate.Q2_Horizon_Graph){
    //     // console.log("Q2_Horizon_Graph_direction", direction)
    //     // console.log("Q2_Horizon_Graphdata", data.data)
    //     // console.log("Q2_Horizon_Graphcolor", data.color)
    //     // console.log("Q2_Horizon_Graphencoding x", data.encoding.x)
    //     // console.log("Q2_Horizon_Graphencoding y", data.encoding.y)

    //     vegaEmbed(`#${svgId}`, data, {
    //       renderer: "svg",
    //       actions: false,
    //     }).then(() => {
    //       let tmp = document.getElementById(svgId);
    //       console.log("Q2_Horizon_GraphgetElementById", tmp);
    //       // let content = document.getElementById(svgId).childNodes[0];
    //       // let width = document.getElementById(svgId).clientWidth;
    //       // let height = document.getElementById(svgId).clientHeight;
    //       // if (
    //       //   content.getBBox().width > width ||
    //       //   content.getBBox().height > height
    //       // ) {
    //       //   console.log("Q2_Horizon_Graph_direction.getBBox().width", content.getBBox().width, width);
    //       //   console.log("Q2_Horizon_Graph_direction.getBBox().height", content.getBBox().height, height);
    //       //   content.setAttribute(
    //       //     "transform",
    //       //     // "translate(" +
    //       //     //   -5 +
    //       //     //   "," +
    //       //     //   -5 +
    //       //     //   ") " + 
    //       //       "scale(" +
    //       //       width / content.getBBox().width +
    //       //       "," +
    //       //       height / content.getBBox().height +
    //       //       ")"
    //       //     );
    //       //   }
    //     });
    //   }
    //   else if (currentTemplate.name === supportedTemplate.NQ_RadialPlot){
    //     vegaEmbed(`#${svgId}`, data, {
    //       renderer: "svg",
    //       actions: false,
    //     }).then(() => {
    //       let content = document.getElementById(svgId).childNodes[0];
    //       console.log("NQ_RadialPlot.content", content);
    //       let width = document.getElementById(svgId).clientWidth;
    //       let height = document.getElementById(svgId).clientHeight;
    //       // width = 150;
    //       // height = 150;
    //       let wScale = width / content.getBBox().width;
    //       let hScale = height / content.getBBox().height;
    //       let scale = wScale > hScale ? hScale : wScale; // Use the smaller one 
    //       console.log("NQ_RadialPlot.getBBox().width", content.getBBox().width, width, wScale);
    //       console.log("NQ_RadialPlot.getBBox().height", content.getBBox().height, height, hScale, scale);
    //       // content.setAttribute("transform", "translate(" + (-5) + "," + -5 + ") scale(" + wScale + "," + hScale + ")");
    //       content.setAttribute("transform", "translate(" + (-5) + "," + -5 + ") scale(" + scale + "," + scale + ")");
    //       // content.setAttribute("transform", "scale(" + scale + "," + scale + ")");
    //     });
    //   }
    //   else if (currentTemplate.name === supportedTemplate.ANQN_Multi_Series_Line_Chart){
    //     vegaEmbed(`#${svgId}`, data, {
    //       renderer: "svg",
    //       actions: false,
    //     }).then(() => {
    //       let content = document.getElementById(svgId).childNodes[0];
    //       console.log("ANQN_Multi_Series_Line_Chart.content", content);
    //       // let width = document.getElementById(svgId).clientWidth;
    //       // let height = document.getElementById(svgId).clientHeight;
    //       // width = 150;
    //       // height = 150;
    //       // let wScale = width / content.getBBox().width;
    //       // let hScale = height / content.getBBox().height;
    //       // let scale = wScale > hScale ? hScale : wScale; // Use the smaller one 
    //       // console.log("ANQN_Multi_Series_Line_Chart.getBBox().width", content.getBBox().width, width, wScale);
    //       // console.log("ANQN_Multi_Series_Line_Chart.getBBox().height", content.getBBox().height, height, hScale, scale);
    //       // content.setAttribute("transform", "translate(" + (-5) + "," + -5 + ") scale(" + wScale + "," + hScale + ")");
    //       content.setAttribute("transform", "translate(" + (-5) + "," + -5 + ")");
    //       // content.setAttribute("transform", "scale(" + scale + "," + scale + ")");
    //     });
    //   }
    //   else{
    //     //all ok
    //     vegaEmbed(`#${svgId}`, data, {
    //       renderer: "svg",
    //       actions: false,
    //     }).then(() => {
    //       let content = document.getElementById(svgId).childNodes[0];
    //       let width = document.getElementById(svgId).clientWidth;
    //       let height = document.getElementById(svgId).clientHeight;
    //       // if (
    //       //   content.getBBox().width > width ||
    //       //   content.getBBox().height > height
    //       // ) {
    //         console.log("curcontent.getBBox().width", content.getBBox().width, width, width / content.getBBox().width);
    //         console.log("curcontent.getBBox().height", content.getBBox().height, height, height / content.getBBox().height);
    //         content.setAttribute(
    //           "transform",
    //           "translate(" +
    //             -5 +
    //             "," +
    //             -5 +
    //             ") " +
    //             "scale(" +
    //             width / content.getBBox().width +
    //             "," +
    //             height / content.getBBox().height +
    //             ")"
    //         );
          
    //     });
    //   }
      
    // });



    this.$bus.$on("embed-id-chat", (id, svgId, visdata, metadata, vis_type, direction)=>{
      // console.log("embed-id-chat", id, svgId, visdata, metadata, vis_type, direction);
      visdata = JSON.parse(visdata);
      metadata = JSON.parse(metadata);

      // if (typeof metadata != Object) {
      //   metadata = JSON.parse(metadata);
      // }
      // console.log("GetTemplate-svgId", svgId)
      // console.log("GetTemplate-vis_type", vis_type)
      // console.log("GetTemplate-metadata", metadata)
      // console.log("GetTemplate-visdata", visdata)
      // console.log("GetTemplate-direction", direction)
      // if (vis_type == supportedTemplate.NQ_Box_Plot || vis_type == supportedTemplate.NQor2Q_Simple_Line_Chart){
      if (vis_type == supportedTemplate.NQ_Box_Plot){
        // let pre_direction = direction;
        if (direction == 'horizon')
          direction = 'vertical';
        else
          direction = 'horizon';
        // console.error("vis_typevis_typeNQ_Box_Plot", pre_direction, direction);
      }
      // else
      //   console.error("vis_typevis_type", vis_type);
      let currentTemplate = GetTemplate(vis_type, metadata, visdata, direction);
      console.log("currentTemplate", vis_type, currentTemplate)
      // console.log("currentTemplate_name", vis_type)

      let height = document.getElementById("vis-panel").clientHeight * 0.23;
      let width = document.body.clientWidth * 0.17;
      let data = JSON.parse(
        JSON.stringify(currentTemplate.GetVegaLite(height, width, true))
      );
      // console.log("currentTemplate", currentTemplate);
      // console.log("embed-id-chat", svgId, currentTemplate.name, data);
      // console.log("hpwidthheight", width, height);
      //others types.GetVegaLite
      // if (vis_type == supportedTemplate.NQ_Box_Plot){
      //   console.log("alter-content.before", width, height);
      // }
      // // console.log("alter-preview-config data", data);
      // // console.log("alter-preview-config height", height);
      // // console.log("alter-preview-config width", width);
      // // console.log("alter-preview-config", data, height, width);
      // if(currentTemplate.name == supportedTemplate.NQor2Q_Simple_Line_Chart){
      //   console.log("GetTemplate-svgId", svgId)
      //   console.log("GetTemplate-vis_type", vis_type)
      //   console.log("GetTemplate-metadata", metadata)
      //   console.log("GetTemplate-visdata", visdata)
      //   console.log("GetTemplate-direction", direction)
      //   console.log("NQor2Q_Simple_Line_Chart", data.encoding);
      //   data.encoding.x.field = "row 2";
      // }
      // if (currentTemplate.name == supportedTemplate.NQ_Box_Plot)
      //   console.log("hp_NQ_Box_Plot", data);


      if(vis_type == "Unit Visualization"){
        // console.log("Unit_data", data.data)
        // console.log("Unit_encoding color", data.encoding.color)
        // console.log("Unit_encoding x", data.encoding.x)
        // console.log("Unit_encoding y", data.encoding.y)
      }
      // else if (vis_type == supportedTemplate.NQ_Box_Plot){
      //   console.log("NQ_Box_Plot_direction", direction)
      //   // if(direction == "horizon")
      //   //   data.encoding.x = [];
      //   // else
      //   //   data.encoding.y = [];
      //   data.encoding.color = [];
      //   data.encoding.x = [];
      //   // console.log("render Box_Plot", data);
      //   // console.log("NQ_Box_Plot_data", data.data)
      //   // console.log("NQ_Box_Plot_color", data.color)
      //   // console.log("NQ_Box_Plot_encoding color", data.encoding.color)
      //   // console.log("NQ_Box_Plot_encoding x", data.encoding.x)
      //   // console.log("NQ_Box_Plot_encoding y", data.encoding.y)

      //   vegaEmbed(`#${svgId}`, data, {
      //     renderer: "svg",
      //     actions: false,
      //   }).then(() => {
      //     let tmp = document.getElementById(svgId);
      //     console.log("tmpgetElementById", tmp);
      //     let content = document.getElementById(svgId).childNodes[0];
      //     let width = document.getElementById(svgId).clientWidth;
      //     let width2 = document.getElementById(svgId).childNodes[0].clientWidth;
      //     let height = document.getElementById(svgId).clientHeight;
      //     console.log("alter-content.chart", width, height);
      //     console.log("alter-content.contend", content, content.getBBox().width, content.getBBox().height);
      //     if (
      //       // content.getBBox().width > width ||
      //       // content.getBBox().height > height
      //       content.clientWidth > width ||
      //       content.clientHeight > height
      //     ) {
      //       console.log("NQ_Box_Plot_direction.getBBox().width", content.getBBox().width, width);
      //       console.log("NQ_Box_Plot_direction.getBBox().height", content.getBBox().height, height);
      //       content.setAttribute(
      //         "transform",
      //         "translate(" + -5 + "," + -5 + ") " + "scale(" +
      //           // width / content.getBBox().width + "," + height / content.getBBox().height + ")"
      //           width / content.clientWidth + "," + height / content.clientHeight + ")"
      //         );
      //       }
      //   });
      // }
      else
        vegaEmbed(`#${svgId}`, data, {
          renderer: "svg",
          actions: false,
        }).then(() => {
          // if (currentTemplate.name == supportedTemplate.ANQN_Multi_Series_Line_Chart){
          //   console.error("svgIdANQN_Multi_Series_Line_Chart", data);
          // }

          if (currentTemplate.name == supportedTemplate.NQ_RadialPlot){
            let content = document.getElementById(svgId);
            content.removeAttribute("transform");
            content.setAttribute("transform", "translate(" + 22 + "," + 22 + ")");
          }
          // let content = document.getElementById(svgId);
          // content.removeAttribute("transform");
          // let content = document.getElementById(svgId);
          // console.log("content", currentTemplate.name, content);
          // if (currentTemplate.name === supportedTemplate.Q2_Horizon_Graph || 
          //   data.mark.type == "area" && currentTemplate.name != supportedTemplate.DensityPlot && currentTemplate.name != supportedTemplate.Histogram_Area) {
          //   content.removeAttribute("transform");

          //   let offset = width / metaData.x.range;
          //   let xScale = width / (width - offset);
          //   content.setAttribute("transform", "translate(" + (-(offset / 2 + 5) * xScale) + "," + -5 + ") scale(" + xScale + ",1)");
          // }
          // else if (currentTemplate.name === supportedTemplate.NQ_PieChart 
          //         || currentTemplate.name === supportedTemplate.NQ_RadialPlot) {
          //   // content.setAttribute("transform", "translate(" + -5 + "," + -5 + ")");
          //   content.setAttribute("transform", "scale(" + 0.9 + "," + 0.9 + ")");
          // }
          // else if (currentTemplate.name === supportedTemplate.Q2_Scatter_plot 
          //         || currentTemplate.name === supportedTemplate.NQ_Histogram_Heatmap 
          //         || currentTemplate.name === supportedTemplate.NQ_Histogram_Scatterplot) {

          //   content.removeAttribute("transform");
          //   content.setAttribute("transform", "translate(" + 10 + "," + 10 + ")");
          // }
          // else if (currentTemplate.name != supportedTemplate.DensityPlot) {
          //   console.log("vegaEmbed != DensityPlot", currentTemplate.name);
          // }
          // else if (content.getBBox().width > width || content.getBBox().height > height) {
          //   let wScale = width / content.getBBox().width;
          //   let hScale = height / content.getBBox().height;
          //   let scale = wScale > hScale ? hScale : wScale; // Use the smaller one 
          //   console.log(content.getBBox());
          //   content.removeAttribute("transform");
          //   content.setAttribute("transform", "translate(" + (-5) + "," + -5 + ") scale(" + scale + "," + scale + ")");
          // }
          
        });
      
    });
  
    this.$bus.$on("alter-preview-config", () => {
      console.log("alter-preview-configalternative_list", this.alternative_list.length, this.showPanelView)
      if (this.showPanelView && this.activeName == 'second') {
          console.log("alterrrrrr", this.alternative_list)
          for(var id in this.alternative_list){
            // let svgId = `alter-chart-${id}`;
            let svgId = `alter_chart_${id}`;
            let insight = this.alternative_list[id];
            console.log("alternative_list.forEach", svgId, id)
            // let height = document.getElementById("vis-panel").clientHeight * 0.25;
            // let width = document.body.clientWidth * 0.19;
 
            // if (insight.vis_type != "Unit Visualization") { // 不是unit vis
            //   var removeZero = false
            //   if (insight.type == 'Skewness' || insight.type == 'Kurtosis') {
            //     removeZero = true
            //   }
            //   let reco_arr = this.generate_vega_recommendation_data(removeZero, insight.rec_list, insight.hp_pos_rec_list)
            //   let config = {'name': insight.vis_type, 'direction': insight.direction, 'encoding': insight.encoding, 'recommend':reco_arr, 'insight':insight, 'hp_pos':hp_pos}
            //   this.auto_create_vega_vis(pos.top, pos.bottom, pos.left, pos.right, config)
            // } 
            // else {  // unit vis需单独讨论
            //   let config = {
            //     //这里写那些需要编码的encodings，目前是固定的，如果hp要改记得在这里改
            //     "name":'unit',"encodings": {"color": true,"height": false,"width": true,"xOffset": false,"yOffset": false,"size": true},"shape": "square","color": "#1783b1","relativeSize": 0.8,"scale": "linear","align": "middle","isrec": false
            //   }
            //   this.auto_create_unit_vis(pos.top, pos.bottom, pos.left, pos.right, config)
            // }
            let supportedTemplate = {
              "unit visualization": "Unit Visualization",
              "line chart": "Line Chart",
              "strip plot": "Strip Plot",
              "box plot": "Box Plot",
              "bar chart": "Bar Chart",
              "horizon graph": "Horizon Graph",
              "scatter plot": "Scatterplot",
              "parallel coordinte plot": "Parallel Coordinate Plot",
              "pie chart": "Pie Chart",
              "multi series line chart": "Multi Series Line Chart",
              "density plot": "Density Plot",
              "radial plot": "Radial Plot",
              "multi line chart": "Multi Series Line Chart",
              // "histogram bar": "Histogram Bar",
              // "histogram area": "Histogram Area",
              "histogram bar": "Density Plot",
              "histogram area": "Density Plot",
            }
            let vis_type = supportedTemplate[insight.vis_type]
            let direction = insight.is_horizontal || insight.vis_type=='Multi Series Line Chart' ? "horizon" : "vertical"

            // let visdata
            // let metadata
            if (vis_type != "Unit Visualization"){
              // console.log("insight.position", insight.hp_pos)
              // visdata, metadata = hp_get_vega_vis(insight.hp_pos[0][0], insight.hp_pos[0][1], insight.hp_pos[1][0], insight.hp_pos[1][1], insight.vis_type)
              console.log(typeof(insight))
              console.log("hp_get_vega_vissend", id, svgId, insight.hp_pos[0][0], insight.hp_pos[0][1], insight.hp_pos[1][0], insight.hp_pos[1][1], vis_type, direction)
              this.$bus.$emit("hp_get_vega_vis", id, svgId, insight.hp_pos[0][0], insight.hp_pos[0][1], insight.hp_pos[1][0], insight.hp_pos[1][1], vis_type, direction)
            }
            // else{
            //   console.log("unit vis null")
            // }
            else{
              // // visdata, metadata = hp_get_unit_vis(insight.hp_pos[0][0], insight.hp_pos[0][1], insight.hp_pos[1][0], insight.hp_pos[1][1])
              // // console.log("hp_get_unit_visvisdata", visdata, metadata)
              
              // this.$bus.$emit("hp_get_unit_vis", id, svgId, insight.hp_pos[0][0], insight.hp_pos[0][1], insight.hp_pos[1][0], insight.hp_pos[1][1], vis_type, direction)
            }
          };
      }
    });

    this.$bus.$on("visualize-recommendData", (array) => {
      this.recommendData_arr = array;
      if (this.recommendData_arr.length > 1) {
        this.$bus.$emit("show-dialog", {
          title: "Batch Operation",
          text: DialogTexts.recommend,
        });
      }
    });

    // User select data
    this.$bus.$on("visualize-selectedData", (position, visData, metaData, hp_pos) => {
      this.update_activeName_first();
      this.figID = "";
      // console.log("new id");
      this.position = position; // for visDatabase to use
      this.hp_pos = hp_pos;
      // console.error("hp_poshp_pos", hp_pos);

      this.visData = JSON.parse(visData);
      this.metaData = JSON.parse(metaData);

      if (typeof metaData != Object) {
        metaData = JSON.parse(metaData);
      }

      if (metaData.x.range == 1 && metaData.y.range == 1) {
        this.unitData_arr = JSON.parse(visData);
        this.OpenUnitView();
      } else {
        this.OpenTemplateView();
      }
      
    });

    // User move table line and modify available space
    this.$bus.$on("rerender-selectedData", (prePosition, afterPosition) => {
      this.VisDB.ReconfigAllCanvas(
        prePosition.x,
        prePosition.y,
        afterPosition.x,
        afterPosition.y
      );
    });

    // User click vis. Restore previous context.
    this.$bus.$on("select-canvas", (id) => {
      this.update_activeName_second();
      this.figID = id;
      this.currentGroupID = this.VisDB.GetGroupID(this.figID);
      if (this.VisDB.database[id].type === "vega") {
        this.currentTemplate = this.VisDB.GetTemplate(id);
        this.visData = this.VisDB.database[id].visData;
        this.metaData = this.VisDB.database[id].metaData;
        this.showPanelView = false;
        this.OpenPanelView();
      } else {
        let group = this.VisDB.GetGroupMembers(this.figID);
        this.unitData_arr = [];
        for (let i = 0; i < group.length; i++) {
          let db = this.VisDB.database[group[i]];
          if (!!db) {
            let config = {};
            config.value = db.visData;
            config.id = db.id;
            config.position = {
              x: db.x,
              y: db.y,
              height: db.height,
              width: db.width,
            };
            this.unitData_arr.push(config);
          }
        }
        this.OpenUnitView();
      }
    });

    // User close a canvas that belongs to a group
    this.$bus.$on("remove-groupCanvas", () => {
      this.$bus.$emit("show-dialog", {
        title: "Batch Operation",
        text: DialogTexts.remove,
      });
    });

    // resize function
    let $bus = this.$bus;
    let resizeTimeout;
    window.addEventListener(
      "resize",
      () => {
        if (!resizeTimeout) {
          resizeTimeout = setTimeout(function () {
            resizeTimeout = null;
            // The actualResizeHandler will execute at a rate of 15fps
            // console.log("preview_config-window.addEventListener");
            $bus.$emit("preview-config");
          }, 66);
        }
      },
      false
    );

    // Background hightlight
    let emitTimeout = undefined;
    let oldHoverText = [];
    let hoverText = []; // the reference should not be inside the call stack of observer. otherwise the popout will not disappear after unhover event happens.

    let isHighlight = false;

    // listen #vg-tooltip-element
    let observer = new MutationObserver(function (mutations, observer) {
      // console.log(mutations);
      mutations.forEach((mutation) => {
        if (mutation.type == "childList") {
          // console.log(mutation);

          if (!emitTimeout) {
            emitTimeout = setTimeout(() => {
              let el = document.getElementById("vg-tooltip-element");

              hoverText = [];
              el.childNodes[0].childNodes[0].childNodes.forEach((childNode) =>
                childNode.childNodes.forEach((value) => {
                  let text = JSON.parse(JSON.stringify(value.textContent));
                  if (value.getAttribute("class") == "key") {
                    hoverText.push(text.substring(0, text.length - 1));
                  } else {
                    hoverText.push(text);
                  }
                })
              );

              let isEq = true;
              if (hoverText.length != oldHoverText.length) {
                isEq = false;
              } else {
                for (let i = 0; i < hoverText.length; i++) {
                  if (oldHoverText[i] != hoverText[i]) {
                    isEq = false;
                    break;
                  }
                }
              }
              if (!isEq) {
                // console.log("hover Text", hoverText);
                if (isHighlight) {
                  $bus.$emit("unhover-field");
                }
                if (
                  document
                    .getElementById("vg-tooltip-element")
                    .getAttribute("class") != ""
                ) {
                  oldHoverText = hoverText;
                  hoverText.forEach((text) => {
                    // not submit headers. but if needed, we can comment the judgement below
                    if (
                      text.substring(0, 3) != "row" &&
                      text.substring(0, 3) != "col"
                    ) {
                      $bus.$emit("hover-field", text);
                    }
                  });
                  isHighlight = true;
                }
              }

              emitTimeout = undefined;
            }, 50);
          }
        } else {
          if (mutation.attributeName == "class") {
            // console.log(
            //   "now class",
            //   document
            //     .getElementById("vg-tooltip-element")
            //     .getAttribute("class")
            // );
            if (
              document
                .getElementById("vg-tooltip-element")
                .getAttribute("class") == ""
            ) {
              if (isHighlight) {
                $bus.$emit("unhover-field");
                isHighlight = false;
                oldHoverText = [];
              }
            }
          }
        }
      });
    });

    observer.observe(document.querySelector("#vg-tooltip-element"), {
      childList: true,
      attributes: true,
      // attributes: true,
    });
  },

  beforeDestroy() {
    this.$bus.$off("preview-config");
    this.$bus.$off("visualize-selectedData");
    this.$bus.$off("rerender-selectedData");
    this.$bus.$off("select-canvas");
    this.$bus.$off("remove-groupCanvas");
    this.$bus.$off("embed-id-chat");
    this.$bus.$off("delete-rl-fig-id");
    this.$bus.$off("hidden-rl-fig-id");
    this.$bus.$off("display-rl-fig-id");
    this.$bus.$off("transmit-recommend-value-to-panel");
    // console.log("confirm-dialog-off");
    this.$bus.$off("confirm-dialog");
    this.$bus.$off("confirm-dialog-hp");
    this.$bus.$off("VisView-change-figID");
  },

  beforeMount() {
    // 自动创建vis，使用固定的visual form/encoding等config
    this.$bus.$on("auto-visualize-data", (position, visData, metaData, RLconfig) => {
      this.figID = "";
      this.position = position; // for visDatabase to use
      this.hp_pos = RLconfig.hp_pos
      console.log("VisView hp_pos", this.hp_pos)
      // if (RLconfig.name !== 'unit')
      //   console.log("VisView hp_pos_rec_list", RLconfig.recommend.hp_pos_rec_list)
      this.visData = JSON.parse(visData);
      this.metaData = JSON.parse(metaData);

      // unit-vis单独讨论
      if (this.metaData.x.range == 1 && this.metaData.y.range == 1) {
        this.unitData_arr = JSON.parse(visData);
        this.ApplyUnitVis2Table(RLconfig)
      }
      // vega-vis部分
      else {  
        this.currentTemplate = new VegaTemplate();
        // console.log("GetTemplateRLconfig", RLconfig.name)
        // console.log("GetTemplatethis.metaData", this.metaData)
        // console.log("GetTemplatethis.visData", this.visData)
        // console.log("GetTemplateRLconfig.direction", RLconfig.direction)
        this.currentTemplate = GetTemplate(RLconfig.name, this.metaData, this.visData, RLconfig.direction);
        // 人工修改encoding细节, density plot先不管!!
        if (RLconfig.name != 'Density Plot' && RLconfig.name != 'Histogram Bar' && RLconfig.name != 'Histogram Area') {
          let EC = new EncodingCompiler(this.vegaConfig.encoding, this.ECSelections)
          let schema = EC.GetSchema()
          for (var item in RLconfig.encoding) {
            if (RLconfig.encoding[item] != null) {
              // if (item == 'color') continue // 有bug 先跳过 记得改
              let val = RLconfig.encoding[item]
              if (typeof(val)==='string' && val.substring(0,6) == 'column') { // hp写的是column 3 实际上应该是column c
                let num = parseInt(val[7])   // 获取列号
                let char = String.fromCharCode(65 + num - 1)
                val = "column " + char 
                val = "column " + char 
                val = "column " + char 
              }

              if(item in schema) {
                schema[item][0].value = val
              }
              else {
                if (item == 'color') {
                  // schema[item] = {"field": val}
                }
              }
            }
          }
          let encoding = EC.GetVegaConfig(schema)
          this.currentTemplate.vegaConfig.encoding = encoding
        }

        // 绘制base区域
        this.ApplyVis2Table(false)
        
        // 绘制推荐区域
        this.recommendData_arr = RLconfig.recommend
        // console.error("GenRecommendFigs-ApplyVis2Table", this.recommendData_arr);
        this.VisDB.GenRecommendFigs(
          this.recommendData_arr,
          this.currentTemplate,
          this.figID
        );
      } 
    });
  }
};
</script>

<style lang="less">
#visview-container {
  position: absolute;
  left: 0%;
  top: 5%;
  width: 100%;
  height: 100%;
  #vis-view {
    position: absolute;
    top: 0%;
    bottom: 0%;
    left: 0%;
    right: 0%;
    background-color: white;
    overflow: hidden;
    .el-form-item {
      margin-top: 2px !important;
      margin-bottom: 2px !important;
    }
  }
  // .panel-view-container {
  //   top: 240px;
  //   bottom: 0%;
  //   left: 0%;
  //   right: 0%;
  // }
}

.role-axis-grid {
  display: none;
}
.role-axis-domain {
  display: none;
}
.background {
  display: none;
}

//因为历史原因，这里的unit不是unit visualization，而是panel visualization
.unit-chart {
  margin: 10px;
  // margin: 0px;
  width: 93%;
  height: 24vh;
  border: 1px solid #dddddd;
  // border: 0px solid #dddddd;
  overflow: hidden;
  // overflow-x: auto;
  // overflow-y: auto;
  svg {
    height: 100%;
    width: 100%;
  }
}

#chart {
  width: 100%;
  height: 25vh;
  overflow: hidden;
  // overflow: auto;
}

.hp-test-chart{
  width: 100%;
  height: 25vh;
  overflow: hidden;
}


.vis-test {
  display: none;
  background-color: white;
  position: absolute;
  left: -300px;
  top: 0px;
  width: 200px;
}

.vis-picture {
  .vis-picture-hButton {
    fill: rgb(90, 156, 248);
    visibility: hidden;
    cursor: pointer;
    &:hover {
      fill: rgb(153, 195, 250);
    }
  }
  &:hover .vis-picture-hButton {
    visibility: visible;
  }

  .vis-picture-mButton {
    fill: rgb(90, 156, 248);
    cursor: pointer;
    visibility: visible;
    &:hover {
      fill: rgb(153, 195, 250);
    }
  }
}


.vis-picture-button {
  cursor: pointer;
  fill: rgb(90, 156, 248);
  &:hover {
    fill: rgb(153, 195, 250);
  }
}
// .unit-chart {
//   margin: 10px;
//   // margin: 0px;
//   width: 93%;
//   height: 24vh;
//   border: 1px solid #dddddd;
//   // border: 0px solid #dddddd;
//   // overflow: hidden;
//   overflow-x: auto;
//   overflow-y: auto;
//   svg {
//     height: 100%;
//     width: 100%;
//   }
// }
.view-container{
  display: flex;
  flex-direction: column;
  // flex-shrink:0;
  // top:80%;
  // height: 100%;
  width: 100%;
  height: 80vh;
  overflow-y: auto;
  .alter-unit-chart {
    cursor: pointer;
    margin: 10px;
    width: 93%;
    height: 24vh;
    // height: 100%;
    border: 1px solid #dddddd;
    // overflow: auto;
    overflow: hidden;
    svg {
      // pointer-events: none;
      width: 100%;
      height: 100%;
      display: block; 
      // height: 24vh;
      // overflow: hidden;
      // overflow: auto;
    }
  }
}

// //1231 version
// .view-container{
//   // display: flex;
//   flex-direction: column;
//   // flex-shrink:0;
//   // height: 100%;
//   // width: 100%;
//   height: 80vh;
//   overflow-y: auto;
//   .alter-unit-chart {
//     cursor: pointer;
//     // margin: 20px;
//     // margin: 1vh;
//     margin: 10px;
//     // width: 24vh;
//     width: 93%;
//     height: 24vh;
//     border: 1px solid #dddddd;
//     overflow: hidden;
//     // overflow-x: auto;
//     // overflow-y: auto;
//     svg {
//       pointer-events: none;
//       // height: 20vh;
//       width: 100%;
//       width: 100%;
//       overflow: auto;
//     }
//   }
// }
.property-text {
  font-size: 14px;
  line-height: 15px;
  margin-top: 7px;
  margin-right: 10px;
  padding-right: 10px;
  margin-left: 5px;
  font-family: "Avenir", Helvetica, Arial, sans-serif;
  color: #606266;
}
#unit-tooltip-element {
  visibility: hidden;
  padding: 8px;
  position: fixed;
  z-index: 1000;
  font-family: sans-serif;
  font-size: 11px;
  border-radius: 3px;
  box-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
  /* The default theme is the light theme. */
  background-color: rgba(255, 255, 255, 0.95);
  border: 1px solid #d9d9d9;
  color: black;
}
#unit-tooltip-element.visible {
  visibility: visible;
}
#unit-tooltip-element h2 {
  margin-top: 0;
  margin-bottom: 10px;
  font-size: 13px;
}
#unit-tooltip-element img {
  max-width: 200px;
  max-height: 200px;
}
#unit-tooltip-element table {
  border-spacing: 0;
}
#unit-tooltip-element table tr {
  border: none;
}
#unit-tooltip-element table tr td {
  overflow: hidden;
  text-overflow: ellipsis;
  padding-top: 2px;
  padding-bottom: 2px;
}
#unit-tooltip-element table tr td.key {
  color: #808080;
  max-width: 150px;
  text-align: right;
  padding-right: 4px;
}
#unit-tooltip-element table tr td.value {
  display: block;
  max-width: 300px;
  max-height: 7em;
  text-align: left;
}
#unit-tooltip-element.dark-theme {
  background-color: rgba(32, 32, 32, 0.9);
  border: 1px solid #f5f5f5;
  color: white;
}
#unit-tooltip-element.dark-theme td.key {
  color: #bfbfbf;
}
</style>
