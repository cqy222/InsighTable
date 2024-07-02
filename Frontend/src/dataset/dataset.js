export function Dataset () {
    this.tabularDataObjList = []
    this.selectedTabularDataObj = null
    this.selectedDF = null

    this.RLDataObjList = []
    // this.DF_list = []
    // this.select_df = null
    this.AlterConsoleDataObjList = []
    this.AlterIncomeDataObjList = []
    this.selectedRLDataObj = null
}

Dataset.prototype = {
    init: function() {
    },
    updateTabularDatasetList: function (processed_tabular_dataobj_list) {
        this.tabularDataObjList = processed_tabular_dataobj_list
        if (this.tabularDataObjList.length > 0) {
            this.selectedTabularDataObj = this.tabularDataObjList[0]
        }
        console.log('selectedTabularDataObj', this.selectedTabularDataObj)
    },
    updateSelectedTabularDataset: function (filename) {
        // set the selectedTabularDataObj
        console.log('updateSelectedTabularDataset', filename)
        var index = -1
        for (var i=0; i<this.tabularDataObjList.length; i++) {
            if (this.tabularDataObjList[i].filename == filename) {
                index = i
                break
            }
        }
        if (index != -1) {
            this.selectedTabularDataObj = this.tabularDataObjList[index]
            console.log('newselectedTabularDataObj', this.selectedTabularDataObj)
        }
    },
    updateSelectedRLDataset: function (filename) {
        // set the selectedTabularDataObj
        var index = -1
        for (var i=0; i<this.RLDataObjList.length; i++) {
            if (this.RLDataObjList[i].filename == filename) {
                index = i
                break
            }
        }
        if (index != -1) {
            this.selectedRLDataObj = this.RLDataObjList[index]
            this.selectedTabularDataObj = this.RLDataObjList[index]
            console.log('newselectedTabularDataObj', this.selectedTabularDataObj)
        }
    },
    updateRefreshedRLDataset: function (RL_dataobj_list) {
        // this.RLDataObjList = RL_dataobj_list
        this.RLDataObjList.concat(RL_dataobj_list)
        if (this.RLDataObjList.length > 0) {
            this.selectedRLDataObj = RL_dataobj_list[0]
            this.selectedTabularDataObj = RL_dataobj_list[0]
            console.log('selectedRLDataObj', this.selectedRLDataObj)
            // this.$bus.$emit("change-current-auto-vis-num", 0)
        // this.selectedTabularDataObj = this.RLDataObjList[this.RLDataObjList.length-1]
        }
    },

    updateSelectDF: function (dataframe) {
        this.selectedDF = dataframe
        console.log('updateSelectDF', this.selectedDF)
    },

    updateUploadData: function (uploadDataObj) {
        this.selectedTabularDataObj = uploadDataObj
        console.log('newUploadObj', this.selectedTabularDataObj)
    },

    updateRLDatasetList: function (RL_dataobj_list) {
        this.RLDataObjList = RL_dataobj_list
        if (this.RLDataObjList.length > 0) {
            this.selectedRLDataObj = this.RLDataObjList[0]
            // this.selectedTabularDataObj = this.RLDataObjList[0]
            console.log('selectedRLDataObj', this.selectedRLDataObj)
        // this.selectedTabularDataObj = this.RLDataObjList[this.RLDataObjList.length-1]
        }
    },

    // updateDFList: function (df_list) {
    //     this.DF_list = df_list
    //     if (this.RLDataObjList.length > 0) {
    //         this.selectedRLDataObj = this.RLDataObjList[0]
    //         // this.selectedTabularDataObj = this.RLDataObjList[0]
    //         console.log('selectedRLDataObj', this.selectedRLDataObj)
    //     // this.selectedTabularDataObj = this.RLDataObjList[this.RLDataObjList.length-1]
    //     }
    // },

    updateAlterRLDatasetList: function (data_name, RL_dataobj_list) {
        if(data_name === 'Console'){
            this.AlterConsoleDataObjList = RL_dataobj_list
        }
        else if(data_name == 'Income'){
            this.AlterIncomeDataObjList = RL_dataobj_list
        }
    },
}