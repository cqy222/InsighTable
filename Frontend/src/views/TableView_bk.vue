<text
                  class="table-cell-text"
                  :x="
                    cal_row_header_position(item[1].value, item[1].times).x +
                    textPaddingX
                  "
                  :y="
                    cal_row_header_position(item[1].value, item[1].times).y +
                    textPaddingY
                  "
                >
                  {{ item[1].value[0] == "&" ? "" : item[1].value }}
                </text>


                <text
                  class="table-cell-text"
                  :key="'text-' + i"
                  :x="
                    cal_column_header_position(item[1].value, item[1].times).x +
                    textPaddingX
                  "
                  :y="
                    cal_column_header_position(item[1].value, item[1].times).y +
                    textPaddingY
                  "
                >
                  {{ item[1].value[0] == "&" ? "" : item[1].value }}
                </text>


                
                <foreignObject 
                class="table-cell-text"
                :key="'text-' + i"
                :x="
                    cal_column_header_position(item[1].value, item[1].times).x
                  "
                  :y="
                    cal_column_header_position(item[1].value, item[1].times).y + 5
                  "
                  :width="
                    cal_column_header_position(item[1].value, item[1].times)
                      .width
                  "
                  :height="
                    cal_column_header_position(item[1].value, item[1].times)
                      .height
                  " style="overflow: hidden;
                        white-space: normal;
                        text-overflow: clip;
                        /* word-wrap: break-word; */
                        ">
                  <!-- <body xmlns="http://www.w3.org/1999/xhtml"> -->
                    <!-- <p>这是一段很长很长很长很长很长很长很长很长很长很长很长很长的文字</p> -->
                    {{ item[1].value[0] == "&" ? "" : item[1].value }}
                    <!-- </body> -->
                </foreignObject>




                <foreignObject 
                class="table-cell-text"
                :key="'text-' + i"
                  :x="
                    cal_row_header_position(item[1].value, item[1].times).x +
                    0
                  "
                  :y="
                    cal_row_header_position(item[1].value, item[1].times).y +
                    5
                  "
                  :width="
                    cal_row_header_position(item[1].value, item[1].times).width
                  "
                  :height="
                    cal_row_header_position(item[1].value, item[1].times).height
                  " style="overflow: hidden;
                        white-space: normal;
                        text-overflow: clip;
                        /* word-wrap: break-word; */
                        ">
                  <!-- <body xmlns="http://www.w3.org/1999/xhtml"> -->
                    <!-- <p>这是一段很长很长很长很长很长很长很长很长很长很长很长很长的文字</p> -->
                    {{ item[1].value[0] == "&" ? "" : item[1].value }}
                    <!-- </body> -->
                </foreignObject>