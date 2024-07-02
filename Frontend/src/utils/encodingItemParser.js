export function parse_encoding_item(item, headerange) {
    let res = {}
    // res结果：{position:, vis_type:, direction:, encoding:, insight:, recommend:, rec_list}
    
    // 获得可视化区域坐标
    var top = headerange.bottom+1
    var bottom = headerange.bottom+1
    var left = headerange.right+1
    var right = headerange.right+1 
    // console.log("parse_encoding_item.item", item)
    if (Array.isArray(item.pos_row)) {  // pos_row是数组，则包括多行
      top += item.pos_row[0]
      bottom += item.pos_row[1] - 1
    } else {
      top += item.pos_row
      bottom += item.pos_row
    }
    if (Array.isArray(item.pos_col)) {  // pos_col是数组，则包括多列
      left += item.pos_col[0]
      right += item.pos_col[1] - 1
    } else {
      left += item.pos_col
      right += item.pos_col
    }

    // 获得可视化类型
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

    // 获得encoding参数
    let encoding = {}
    for (var attr in item) {
      // encoding部分不包括下边这几个属性
      if (attr == "pos_col" || attr == "pos_row" || attr == "vis_type" || attr == "insight_type" || attr == "insight_value" || attr == "is_horizontal" || attr == "rec_col_priority" || attr == "rec_col_type" || attr == "rec_row_priority" || attr == "rec_row_type" || attr == "rec_list") continue
      encoding[attr] = item[attr]
    }

    res.position = {'top':top, 'bottom':bottom, 'left':left, 'right':right}
    res.vis_type = supportedTemplate[item.vis_type]
    // res.vis_type = "Pie Chart"
    res.direction = item.is_horizontal || res.vis_type=='Multi Series Line Chart' ? "horizon" : "vertical"
    res.encoding = encoding

    if(item.insight_type == 'Pearsonr') {
      item.insight_type = 'Pearson'
    }
    else if(item.insight_type == 'M-Top 2') {
      item.insight_type = 'Top 2'
    }
    else if(item.insight_type == 'M-Dominance') {
      item.insight_type = 'Dominance'
    }
    else if(item.insight_type == 'M-Evenness') {
      item.insight_type = 'Evenness'
    }
    res.insight = {'type':item.insight_type, 'value':item.insight_value}
    res.recommend = {
      'row': {'type':item.rec_row_type, 'range':[0, item.rec_row_priority]},
      'col': {'type':item.rec_col_type, 'range':[0, item.rec_col_priority]}
    }
    
    res.hp_pos_rec_list = []
    let tmp_pos_row = item.pos_row;
    let tmp_pos_col = item.pos_col;
    // 一个数 或者 开
    if (typeof tmp_pos_row === 'number')
      tmp_pos_row = [tmp_pos_row, tmp_pos_row];
    else
      tmp_pos_row = [tmp_pos_row[0], tmp_pos_row[1]-1];
    if (typeof tmp_pos_col === 'number')
      tmp_pos_col = [tmp_pos_col, tmp_pos_col];
    else
      tmp_pos_col = [tmp_pos_col[0], tmp_pos_col[1]-1];
    item.hp_pos = [tmp_pos_row, tmp_pos_col];
    // console.log("itempos_rowpos_col", item.pos_row, item.pos_col)
    if (item.rec_list!=null && item.rec_list.length > 0) {
      for (var i = 0; i < item.rec_list.length; i++) { 
        // 闭
        let x1 = item.rec_list[i][0];
        let x2 = item.rec_list[i][1];
        let y1 = item.rec_list[i][2];
        let y2 = item.rec_list[i][3];
        // console.log("itempos_rowpos_corec", item.rec_list)
        if (tmp_pos_row[0] === x1 && tmp_pos_row[1] === x2 && tmp_pos_col[0] === y1 && tmp_pos_col[1] === y2){
          console.log("tmp_pos_rowtmp_pos_row")
          continue;
        }
        res.hp_pos_rec_list.push([[x1, x2], [y1, y2]]);
      }
      // console.log("res.hp_pos_rec_list", res.hp_pos_rec_list, 'pos_row', tmp_pos_row, 'pos_col', tmp_pos_col)
    }
    
    res.rec_list = item.rec_list
    if (item.rec_list==null || item.rec_list.length==0) {
      res.rec_list = [[top-headerange.bottom-1, bottom-headerange.bottom-1, left-headerange.right-1, right-headerange.right-1, true]]
    }
    // console.log("parser res", res)

    return res
}