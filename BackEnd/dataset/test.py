# import pandas as pd
# cur_dataframe = pd.read_excel('public/Console Sales.xlsx', header=[0, 1], index_col=[0, 1, 2]).sort_index(axis=0).sort_index(axis=1)
# cur_dataframe *= 0.
# cur_dataframe.iloc[0:0, 0:10] = 1
# print(cur_dataframe.index)
# pos = cur_dataframe.index.get_loc("Microsoft")
# print("cur_dataframe", pos)
pos_row = 0
pos_row = slice(pos_row, pos_row + 1, None) if isinstance(pos_row, int) else pos_row
print(pos_row)