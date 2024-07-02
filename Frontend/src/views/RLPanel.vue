<template>
    <div class="rl-container">
        <div id="el-icon-grid">
        <!-- <div style="margin-left: 15px"> -->
            <el-button  icon="el-icon-s-opportunity" @click="regenerate_list" size="large" id="el-icon-s-opportunity"></el-button>
        <!-- </div> -->
        <!-- <div style="margin-left: 10px"> -->
            <el-button  icon="el-icon-delete" @click="clear_list" size="large" id="el-icon-delete"></el-button>
        <!-- </div> -->
        

        <!-- <div style="margin-left: 15px"> -->
            <el-button  icon="el-icon-arrow-left" @click="change_curr_num_butt(-1)" size="mini" id="el-icon-arrow-left"></el-button>
            <!-- <el-button @click="change_curr_num_butt(1)" size="mini"><i class="el-icon-arrow-right el-icon--right"></i></el-button> -->
        <!-- </div> -->

        <!-- <div style="margin-left: 5px"> -->
            <el-button  icon="el-icon-arrow-right" @click="change_curr_num_butt(1)" size="mini" id="el-icon-arrow-right"></el-button>
            <!-- <el-button @click="change_curr_num_butt(1)" size="mini"><i class="el-icon-arrow-right el-icon--right"></i></el-button> -->
        <!-- </div> -->
        </div>
        
        
        <!-- el-icon-star-off -->
        <!-- el-icon-user -->
        <!-- el-icon-s-opportunity -->
        
        <!-- 每个insight太长了，再短一点 -->
        <div class='step_container'>
            <div :style=el_step_style>
                <el-steps :active="currAutoVisNum" align-center >
                    <el-step
                        @click.native="change_curr_num(-1)"
                        title="0" icon="el-icon-star-off">
                    </el-step>
                    <!-- <el-step v-for="(item, index) in insightList" 
                        :key = index
                        @click.native="change_curr_num(index)"
                        :title="(index+1).toString()" :description="item"
                                        icon="el-icon-user">
                    </el-step> -->
                    <!-- <el-step v-for="(item, index) in insightList"  -->
                    <el-step v-for="(item, index) in insightList_simpname" 
                        :key = index
                        @click.native="change_curr_num(index)"
                        :title="(index+1).toString()" :description="item"
                            :icon="insight_is_user[index] ? 'el-icon-user' : 'el-icon-star-off'">
                    </el-step>
                </el-steps>
            </div>
        </div>
        

    </div>
</template>

<script>
import { EncodingCompiler } from './vis/SchemaCompiler';

export default {
  name: "RLPanel",
  props: ["insightList"],//这里只是insight name
  data() {
    return {
        currAutoVisNum : 0,
        el_step_style: "width: 80px",
        insight_is_user: [],
        insight_both_list: [],
        insightList_simpname: []
    }
  },
  created() {
    // var insight_len = this.insightList.length
    // for (_ in insight_len)
    //     this.insight_is_user.push(false);
    // this.el_step_style = "width: " + insight_len*80 + "px";
    // console.log("createdRLPanel", insight_len);
  },
  mounted() {
    this.currAutoVisNum = 0;

    this.$bus.$on("RLPanel-visualize-next-encoding", (next_encoding) => {
        //change next_encoding add to -> this.insightList & this.insight_both_list
        console.log("RLPanel-visualize-next-encoding", next_encoding, typeof next_encoding);
        let type_mp = {
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
        
        for (var encoding of next_encoding){
            let cur_type = encoding.vis_type
            if (cur_type in type_mp)
                cur_type = type_mp[cur_type]
            this.insightList.push(cur_type);

            if (encoding.hp_pos == null || encoding.hp_pos == undefined){
                encoding.hp_pos = this.pos_rowcol_2_hp_pos(encoding.pos_row, encoding.pos_col);
                console.error("Null value in encoding", encoding.hp_pos);
            }
            var selectedArea = {"top":encoding.hp_pos[0][0], "bottom":encoding.hp_pos[0][1], "left":encoding.hp_pos[1][0], "right":encoding.hp_pos[1][1]}
            var recommendArea = [];
            if (encoding.rec_list != null)
                for (var rec of encoding.rec_list){
                    recommendArea.push({"top":rec[0], "bottom":rec[1], "left":rec[2], "right":rec[3]})
                }
            this.insight_both_list.push({"selectedArea":selectedArea,
                    "recommendArea":recommendArea, "insight_type":cur_type});

            this.insight_is_user.push(false);
            // console.log("encoding of next_encoding", encoding);
            this.currAutoVisNum += 1;
        };
    });
    
    this.$bus.$on("RLPanel-add-user-icon", (insight_type)=>{
        // hp_pos
        // insight_type
        // rec_list[(pos_row, pos_col)]
        this.insight_is_user = this.insight_is_user.slice(0, this.currAutoVisNum);
        this.$bus.$emit("clear-insightList", this.currAutoVisNum, insight_type);
        this.$bus.$emit("clear-currentFigIDList", this.currAutoVisNum, insight_type);

        this.tmp_insight_type = insight_type;
        this.insight_is_user.push(true);
        // this.insightList.push(insight_type);
        console.log("RLPanel-add-user-icon", insight_type, this.insightList.length, this.insightList);
        this.el_step_style = "width: " + (this.insightList.length+1)*80 + "px";
        this.currAutoVisNum += 1
        console.log("RLPanel-add-user-icon.currAutoVisNum", this.currAutoVisNum);
        // console.log("RLPanel-add-user-icon", insight_type);
        // this.$bus.$emit("TableView-add-user-to-list", insight_type);//放在clear-currentFigIDList里面了，避免先后运行顺序不同的问题
    });
    this.$bus.$on("RLPanel-add-user-icon_reclist", (selectedArea, recommendArea)=>{
        this.insight_both_list = this.insight_both_list.slice(0, this.currAutoVisNum-1);//-1是因为RLPanel-add-user-icon已经对其+=1了

        // top bottom left right
        console.log("RLPanel-add-user-icon_reclist", this.insightList, this.insightList[this.insightList.length - 1], selectedArea, recommendArea);
        if (this.tmp_insight_type == null || this.tmp_insight_type == undefined)
            console.error("RLPanel-add-user-icon_reclist-null-tmp_insight_type", this.tmp_insight_type);
        this.insight_both_list.push({"selectedArea":selectedArea, "recommendArea":recommendArea, "insight_type":this.tmp_insight_type});//for re generate
    });
    this.$bus.$on("RLPanel-add-user-icon-unit", (insight_type, selectedArea, recommendArea)=>{
        // hp_pos
        // insight_type
        // rec_list[(pos_row, pos_col)]
        console.log("RLPanel-add-user-icon-unit", insight_type, selectedArea, recommendArea);
        this.insight_is_user = this.insight_is_user.slice(0, this.currAutoVisNum);
        this.$bus.$emit("clear-insightList", this.currAutoVisNum, insight_type);
        this.$bus.$emit("clear-currentFigIDList", this.currAutoVisNum, insight_type);

        this.insight_is_user.push(true);
        // this.insightList.push(insight_type);
        // console.log("RLPanel-add-user-icon", insight_type, this.insightList.length, this.insightList);
        this.el_step_style = "width: " + (this.insightList.length+1)*80 + "px";
        this.currAutoVisNum += 1
        // console.log("RLPanel-add-user-icon.currAutoVisNum", this.currAutoVisNum);

        this.insight_both_list = this.insight_both_list.slice(0, this.currAutoVisNum-1);//-1是因为RLPanel-add-user-icon已经对其+=1了
        // console.log("RLPanel-add-user-icon_reclist", this.insightList, this.insightList[this.insightList.length - 1], selectedArea, recommendArea);
        this.insight_both_list.push({"selectedArea":selectedArea, "recommendArea":recommendArea, "insight_type": "Unit"});//for re generate
    });
    this.$bus.$on("RLPanel-remove-pos", (pos)=>{
        console.log("RLPanel-remove-pos", pos);
        console.log("RLPanel-remove-pos.insight_both_list", this.insight_both_list.length);
        console.log("RLPanel-remove-pos.insightList", this.insightList.length);
        console.log("RLPanel-remove-pos.insight_is_user", this.insight_is_user.length);
        var new_insight_both_list = [];
        var new_insightList = [];
        var new_insight_is_user = [];
        for (let cur_p in this.insight_both_list){
            var cur_insight_both_list = this.insight_both_list[cur_p];
            var cur_insightList = this.insightList[cur_p];
            var cur_insight_is_user = this.insight_is_user[cur_p];
            if (cur_p != pos){
                new_insight_both_list.push(cur_insight_both_list);
                new_insightList.push(cur_insightList);
                new_insight_is_user.push(cur_insight_is_user);
            }
        }
        this.insight_both_list = new_insight_both_list;
        this.$bus.$emit("App-alter_update_vis-end", new_insightList);
        this.insight_is_user = new_insight_is_user;
        this.currAutoVisNum = new_insight_both_list.length;
    });

    this.$bus.$on("RLPanel-alter_update_vis_on", (item)=>{
        console.log("RLPanel-alter_update_vis_on", item);
        this.$bus.$emit("TableView-alter_update_vis_on", item, this.insightList, this.insight_is_user, this.insight_both_list);
    });

    this.$bus.$on("RLPanel-alter_update_vis-end", (new_insight_is_user, new_insight_both_list)=>{
        this.insight_is_user = new_insight_is_user;
        this.insight_both_list = new_insight_both_list;
        this.currAutoVisNum = new_insight_both_list.length;
        console.log("RLPanel-alter_update_vis-end", this.insight_is_user, this.insight_both_list);
    });
  },
  methods: {
    pos_rowcol_2_hp_pos(pos_row, pos_col){
        let tmp_pos_row = []
        let tmp_pos_col = []
        if (typeof pos_row === 'number')
            tmp_pos_row = [pos_row, pos_row];
        else
            tmp_pos_row = [pos_row[0], pos_row[1]-1];
        if (typeof pos_col === 'number')
            tmp_pos_col = [pos_col, pos_col];
        else
            tmp_pos_col = [pos_col[0], pos_col[1]-1];
        let hp_pos = [tmp_pos_row, tmp_pos_col];
        return hp_pos;
    },
    change_curr_num(index) {
        console.log("change-curr-num", index)
        this.currAutoVisNum = index+1
        this.$bus.$emit("change-current-auto-vis-num", index)
    },
    change_curr_num_butt(num) {
        if (this.currAutoVisNum == 0 && num == -1 || this.currAutoVisNum == this.insightList.length && num == 1) return
        this.currAutoVisNum += num
        this.$bus.$emit("change-current-auto-vis-num", this.currAutoVisNum-1)
    },
    regenerate_list(){
        this.insight_both_list = this.insight_both_list.slice(0, this.currAutoVisNum);
        this.insight_is_user = this.insight_is_user.slice(0, this.currAutoVisNum);
        console.log("currAutoVisNumcurrAutoVisNum", this.currAutoVisNum);
        console.log("erererereinsightList", this.insightList.length, this.insightList);
        console.log("rerererereinsight_both_list", this.insight_both_list.length, this.insight_both_list);
        console.log("rerererereinsight_is_user", this.insight_is_user.length, this.insight_is_user);
        this.$bus.$emit("clear-insightList", this.currAutoVisNum);
        this.$bus.$emit("clear-currentFigIDList", this.currAutoVisNum);
        this.$bus.$emit("update_next_encoding", this.insight_both_list);
        // console.error("null_insightList-regenerate_list", this.insightList.length);
    },
    clear_list(){
        this.$bus.$emit("change-current-auto-vis-num", -1);
        this.currAutoVisNum = 0;
        this.insight_both_list = [];
        this.insight_is_user = [];
        this.$bus.$emit("clear-insightList");
        this.$bus.$emit("clear-currentFigIDList");
    }
  },
  watch: {
    insightList(newVal, oldVal) {
        var insight_len = newVal.length
        for (_ in insight_len)
            this.insight_is_user.push(false);
        if (this.insight_is_user.length != newVal.length)
            console.error("this.insight_is_user.length != this.insightList.length");
        this.el_step_style = "width: " + (insight_len+1)*80 + "px";
        console.trace("insightList_changed", newVal);
        // console.error("null_insightList-watch", insight_len);
        // console.log("this.el_step_style", this.el_step_style);\
        
        let type_mp = {
            "Unit Visualization":'Outliers',
            "Line Chart":'Change Point',
            // "Strip Plot":,
            "Box Plot":'Outliers',
            "Bar Chart":'Outliers',
            "Horizon Graph":'Trend',
            // "Scatterplot":,
            "Parallel Coordinate Plot":'Correlation',
            "Pie Chart":'Dominance',
            "Multi Series Line Chart":'Correlation',
            "Density Plot":'Skewness',
            "Radial Plot":'Dominance',
        }
        var tmp_list = [];
        for (let cur_type of newVal){
            if (cur_type in type_mp)
                cur_type = type_mp[cur_type]
            tmp_list.push(cur_type);
        }
        this.insightList_simpname = tmp_list;
    },
  },
  beforeDestroy(){
    this.$bus.$off("RLPanel-add-user-icon");
    this.$bus.$off("RLPanel-add-user-icon_reclist");
    this.$bus.$off("RLPanel-visualize-next-encoding");
    this.$bus.$off("RLPanel-add-user-icon-unit");
    this.$bus.$off("RLPanel-alter_update_vis_on");
    this.$bus.$off("RLPanel-alter_update_vis-end");
    this.$bus.$off("RLPanel-remove-pos");
  }
}
</script>

<style scoped lang="less">
#el-icon-grid{
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 10px; /* 可选的间距 */
    margin-left: 10px;
}
#el-icon-s-opportunity{
    margin-left: 5px;
    margin-top: 0px;
    padding: 8px 8px;
    font-size: 18px;
}
#el-icon-delete{
    margin-left: 5px;
    margin-top: 0px;
    padding: 8px 8px;
    font-size: 18px;
}
#el-icon-arrow-left{
    margin-left: 5px;
    margin-top: 0px;
    padding: 8px 8px;
    font-size: 18px;
}
#el-icon-arrow-right{
    margin-left: 5px;
    margin-top: 0px;
    padding: 8px 8px;
    font-size: 18px;
}
// #el-icon-s-opportunity{
//     margin-left: 15px;
//     margin-top: 10px;
//     padding: 8px 8px;
//     font-size: 22px;
// }
// #el-icon-delete{
//     margin-left: 15px;
//     margin-top: 10px;
//     padding: 8px 8px;
//     font-size: 22px;
// }
// #el-icon-arrow-left{
//     margin-left: 15px;
//     margin-top: 10px;
//     padding: 8px 8px;
//     font-size: 22px;
// }
// #el-icon-arrow-right{
//     margin-left: 15px;
//     margin-top: 10px;
//     padding: 8px 8px;
//     font-size: 22px;
// }
.rl-container {
    margin-top:0.8%;
    display: flex;
}
.step_container{
    position: absolute;
    left: 100px;
    right: 1%;
    top: 27px;
    bottom: 1%;

    overflow-x: auto;
    overflow-y: hidden;
}
.el-step{
    cursor: pointer;
    user-select: none;
}
</style>