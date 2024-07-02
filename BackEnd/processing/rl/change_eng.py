import pickle
import sys
import pandas as pd
from visEncoding import VisEncoding

dictionary = {'当年': 'Current year', '当月': 'Current month', '当日': 'Current day', '保费合计': 'Total amount', '同比增速': 'Growth rate',
    '中短存续期': 'Short-term', '其他长险趸交': 'Medium-term','短期险': 'Long-term', '长险趸交':'Others',
    '个险渠道': 'Individual', '互动渠道': 'Interactive', '服务营销渠道': 'Service marketing', '团险渠道': 'Group',
    '电商渠道': 'E-commerce', '银邮渠道': 'Bank',
    '大个险':'Major insurance','非大个险':'Non-major insurance', 
    '内蒙古分公司':'Province 1', '北京市分公司':'Province 2', '吉林省分公司':'Province 3', '天津市分公司':'Province 4', '山西省分公司':'Province 5','河北省分公司':'Province 6', '辽宁省分公司':'Province 7', '黑龙江分公司':'Province 8'}

def get_eng_dict(index):
    res = {}
    for i in range(len(index)):
        before = index[i]
        after = []
        for j in range(len(index[i])):
            ch = index[i][j]
            if ch in dictionary:
                after.append(dictionary[ch])
            else:
                print(ch, 'not in dictionary!')
        after = tuple(after)
        res[before] = after
    return res

if __name__ == '__main__':
    # 获取命令行参数
    args = sys.argv
    # 输出所有命令行参数
    file_name = args[1]
    # read
    with open ("../../public/RL/0326data/"+file_name+".txt", 'rb') as f:
        table, encoding_list = pickle.load(f)
        # new_index = get_eng_dict(table.index)
        # new_column = get_eng_dict(table.columns)

        new_table = table.rename(index=dictionary, columns=dictionary)
        print(new_table)
    
    # write
    with open("../../public/RL/final/"+file_name+".txt", 'wb') as f:
        pickle.dump((new_table, encoding_list), f)
    
   
    


    

