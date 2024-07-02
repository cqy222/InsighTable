import pickle
from visEncoding import VisEncoding

#read
with open ("../../public/RL/0326data/032704.txt", 'rb') as f:
    table, encoding_list = pickle.load(f)
    table.to_csv('test.csv')
    print('table', table)
    print('table.index', table.index)
    print('table.columns', len(table.columns[0]))
    for index in range(len(encoding_list)):
        i = encoding_list[index]
        print('------------------- New VisEncoding', index+1, '-------------------')
        print('vis_type', i.vis_type)
        print('insight_type', i.insight_type)
        print('insight_value', i.insight_value)
        print('pos_row', i.pos_row)
        print('pos_col', i.pos_col)
        print('x', i.x)
        print('y', i.y)
        print('size', i.size)
        print('color', i.color)
        print('shape', i.shape)
        print('align', i.align)
        print('is_horizontal', i.is_horizontal)
        print('theta', i.theta)
        print('radius', i.radius)
        print('scale', i.scale)
        print('rec_row_type', i.rec_row_type)
        print('rec_row_priority', i.rec_row_priority)
        print('rec_col_type', i.rec_col_type)
        print('rec_col_priority', i.rec_col_priority)
        print('rec_list', i.rec_list)
    
    