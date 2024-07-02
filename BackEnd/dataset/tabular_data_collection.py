from processing.tabular_data_parse import parse_sheet
from processing.json_encoder import NpEncoder
from visEncoding import VisEncoding
import pickle
import json, os
import pandas as pd

tabular_dataset_list = []
rl_dataset_list = []
dataframe_list = []
cur_dataframe = None
alternative_dataset_list_Console = []
alternative_dataset_list_Income = []

def load_tabular_dataset():
    '''
        read tabular dataset and process
    '''
    global tabular_dataset_list
    # namelist = ["Console Sales.xlsx", "Console Sales(cumulative).xlsx", "US Investment Abroad.xlsx", "School Curriculums.xlsx"]
    # rowlist = [42,42,41,17]
    # collist = [35,39,15,11]
    namelist = ["Console Sales.xlsx", "保费收入分析.xlsx"]
    rowlist = [42,59]
    collist = [23,24]
    for index in range(len(namelist)):
        # prefix = "web/BackEnd/public/"
        prefix = "public/"
        sheet = "Sheet1"
        filename = prefix + namelist[index]
        s = parse_sheet(filename)
        tabular_data_content = s.result()
        tabular_data_obj = {}
        tabular_data_obj["filename"] = namelist[index]
        tabular_data_obj["row"] = rowlist[index]
        tabular_data_obj["column"] = collist[index]
        # tabular_data_obj["content"] = str(tabular_data_content)
        tabular_data_obj["content"] = tabular_data_content

        json_tabular_data_obj = json.dumps(tabular_data_obj)
        tabular_dataset_list.append(json_tabular_data_obj)

        # tabular_dataset_list的构成：[jsonstr, jsonstr, jsonstr...]
    tabular_dataset_list = json.dumps(tabular_dataset_list)
    
    global cur_dataframe
    # cur_dataframe = pd.read_excel('public/Console Sales.xlsx', header=[0, 1], index_col=[0, 1, 2]).sort_index(axis=0).sort_index(axis=1)
    cur_dataframe = pd.read_excel('public/Console Sales.xlsx', header=[0, 1], index_col=[0, 1, 2])
    # raise NotImplementedError
    # print("cur_dataframe", cur_dataframe)

def load_rl_dataset():
    '''
        读取rl处理结果
    '''
    global rl_dataset_list
    global dataframe_list
    # global cur_dataframe
    # namelist = ["test10","test11", "test12", "test13", "test14", "test15", "test16", "test17", "test18", "test19", "test20","test21","test22","test23"]
    # header_row = [[0, 1, 2], [0, 1, 2], [0,1], [0, 1, 2], [0,1], [0,1],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2],[0, 1, 2,3],[0, 1, 2],[0, 1, 2,3],]   # 行表头的序号范围
    # namelist = ["0323-1", "0323-2", "0323-3", "0323-4", "0323-5", "0323-6", "0323-7", "0323-8", "0323-9", ]
    namelist = ["premium statistics", "032701", "032702", "032703"]
    # header_row = [[0, 1], [0, 1], [0,1,2,3], [0, 1], [0,1], [0,1,2,3],[0, 1,2,3],[0, 1,2,3],[0, 1,2,3]] 
    for index in range(len(namelist)):
        # prefix = "web/BackEnd/public/RL/final/"
        prefix = "public/RL/final/"
        # sheet = "Sheet1"
        filename = prefix + namelist[index] + ".txt"

        # 读取harpoon的结果，返回为一个pandas dataframe格式数据，以及一个encoding结果
        with open (filename, 'rb') as f:
            dataframe, encoding_list = pickle.load(f)
            dataframe_list.append(dataframe)
        # print('dataframe', type(dataframe))
        # 处理pandas表格数据
        header_row_num = len(dataframe.columns[0])
        # processed_filename = 'web/BackEnd/public/RL/pandas_' + namelist[index] + ".xlsx"
        processed_filename = 'public/RL/pandas_' + namelist[index] + ".xlsx"
        dataframe.to_excel(processed_filename) # 过渡阶段 没写好新的数据结构 暂时这么实现（把pandas表格转化为xlsx再用原本的函数读取）
        s = parse_sheet(processed_filename, header_row_num )
        tabular_content = s.result()   # 处理好的表格数据

        # 处理encoding结果
        encoding_res = []
        attributes = [attr for attr in dir(encoding_list[0]) if not attr.startswith("__")] # encoding list包含的所有属性名
        for item in encoding_list:
            obj = create_encoding_obj(item, attributes)
            encoding_res.append(obj)
        
        rl_obj = {}
        rl_obj["filename"] = namelist[index]
        # rl_obj["table"] = str(tabular_content)
        rl_obj["content"] = tabular_content
        rl_obj["encoding"] = encoding_res
        
        json_rl_obj = json.dumps(rl_obj, cls=NpEncoder)
        rl_dataset_list.append(json_rl_obj)

        # rl_dataset_list的构成：[jsonstr, jsonstr, jsonstr...]
        os.remove(processed_filename)
        
    rl_dataset_list = json.dumps(rl_dataset_list)

def create_encoding_obj(item, attributes):
    res = {}
    for attr in attributes:
        value = getattr(item, attr) 
        if isinstance(value, slice) : # 是slice类型，需要特殊处理
            value = [value.start, value.stop]
        if attr == "rec_list" and value != None: # 是rec_list 需要特殊处理
            tmp_res = []
            for i in range(0, len(value)):
                if isinstance(value[i][0][0], slice):
                    top = value[i][0][0].start
                    bottom = value[i][0][0].stop-1
                else:
                    top = value[i][0][0]
                    bottom = value[i][0][0]
                
                if isinstance(value[i][0][1], slice):
                    left = value[i][0][1].start
                    right = value[i][0][1].stop-1
                else:
                    left = value[i][0][1]
                    right = value[i][0][1]
                
                tmp_rec = [top, bottom, left, right, value[i][1]]
                # tmp_rec的顺序是top, bottom, left, right, marker
                tmp_res.append(tmp_rec)
            value = tmp_res
        res[attr] = value

    return res

def get_tabular_dataset():
    '''
        read tabular dataset and process
    '''
    # return {"data": str(tabular_dataset_list)}
    return tabular_dataset_list

def get_rl_dataset():
    '''
        read tabular dataset and process
    '''
    return rl_dataset_list

def get_dataframe_list():
    return dataframe_list

def get_dataframe():
    return cur_dataframe

def parse_upload_data(filename):
    # filepath = "web/BackEnd/public/upload_" + filename
    filepath = "public/upload_" + filename

    # # 调用预处理函数，获得初始行列表头范围，现在是手动设置的
    # header_row = [0,1]
    # header_col = [0,1,2]

    # # 加入harpoon的处理函数，返回为一个pandas dataframe格式数据
    # dataframe = harpoonProcess(filepath, header_row, header_col).process()
    # fp = 'public/pandas_' + filename
    # dataframe.to_excel(fp) # 过渡阶段 没写好其他的暂时这么实现
 
    # s = parse_sheet(fp, header_row)

    s = parse_sheet(filepath)
    tabular_data_content = s.result()
    tabular_data_obj = {}
    tabular_data_obj["filename"] = "upload_" + filename
    tabular_data_obj["row"] = 0
    tabular_data_obj["column"] = 0
    # tabular_data_obj["content"] = str(tabular_data_content)
    tabular_data_obj["content"] = tabular_data_content
    
    json_tabular_data_obj = json.dumps(tabular_data_obj)
    tabularlist = json.dumps([json_tabular_data_obj])

    # os.remove(fp)
    # return {"data": str(tabularlist)}
    return tabularlist

# def load_pandas_data(dataframe, name):
#     # dataframe为pandas型数据

#     # 过渡阶段 没写好其他的暂时这么实现
#     filepath = '../public/pandas' + name + '.xlsx'
#     dataframe.to_excel(filepath) 
#     s = parse_sheet(filepath)
#     tabular_data_content = s.result()
#     tabular_data_obj = {}
#     tabular_data_obj["filename"] = "pandas_" + name
#     tabular_data_obj["row"] = 0
#     tabular_data_obj["column"] = 0
#     tabular_data_obj["content"] = str(tabular_data_content)

#     tabularlist = [tabular_data_obj]
#     return {"data": str(tabularlist)}


# def load_refresh_dataset(filename):
#     '''
#         读取rl处理结果
#     '''
#     # prefix = "public/RL/final/"
#     # 读取harpoon的结果，返回为一个pandas dataframe格式数据，以及一个encoding结果
#     # filename = "xxxx.txt"
#     with open (filename, 'rb') as f:
#         dataframe, encoding_list = pickle.load(f)

#     # 处理pandas表格数据
#     header_row_num = len(dataframe.columns[0])
#     # processed_filename = 'public/RL/pandas_' + namelist[index] + ".xlsx"
#     processed_filename = "tmp.xlsx"
#     dataframe.to_excel(processed_filename) # 过渡阶段 没写好新的数据结构 暂时这么实现（把pandas表格转化为xlsx再用原本的函数读取）
#     s = parse_sheet(processed_filename, header_row_num )
#     tabular_content = s.result()   # 处理好的表格数据

#     # 处理encoding结果
#     encoding_res = []
#     attributes = [attr for attr in dir(encoding_list[0]) if not attr.startswith("__")] # encoding list包含的所有属性名
#     for item in encoding_list:
#         obj = create_encoding_obj(item, attributes)
#         encoding_res.append(obj)
    
#     rl_obj = {}
#     rl_obj["filename"] = processed_filename
#     # rl_obj["table"] = str(tabular_content)
#     rl_obj["content"] = tabular_content
#     rl_obj["encoding"] = encoding_res
    
#     json_rl_obj = json.dumps(rl_obj, cls=NpEncoder)
#     rl_dataset_list.append(json_rl_obj)

#     # rl_dataset_list的构成：[jsonstr, jsonstr, jsonstr...]
#     os.remove(processed_filename)
        
#     rl_dataset_list = json.dumps(rl_dataset_list)
import time
import sys
import numpy as np
# sys.path.append("/data2/hp/eval_clean")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
# from ppo_table import eval_result
def get_refresh_dataset(data_name):
    raise NotImplementedError
    # namelist = ["premium statistics", "032701", "032702", "032703"]
    # header_row = [[0, 1], [0, 1], [0,1,2,3], [0, 1], [0,1], [0,1,2,3],[0, 1,2,3],[0, 1,2,3],[0, 1,2,3]] 
    # for index in range(len(namelist)):
    # prefix = "public/RL/final/"
    # sheet = "Sheet1"
    # filename = prefix + namelist[index] + ".txt"
    global cur_dataframe
    if not isinstance(data_name, str):
        print(data_name)
        raise NotImplementedError
    if data_name == '保费收入分析.xlsx':
        data_name = 'Income'
    elif data_name == 'Console Sales.xlsx':
        data_name = 'Console'
    # print('data_name', data_name)
    # raise NotImplementedError
    # print('hpgggggg', os.getcwd())
    path = 'eval_results/' + data_name
    # raise NotImplementedError
    os.makedirs(path, exist_ok=True)
    os.makedirs('eval_results_trash/' + data_name, exist_ok=True)
    files = os.listdir(path)
    refresh_dataset_list = []
    # if len(files) == 0:
    #     eval_result(data_name)
    files = os.listdir(path)
    # print('files', path, len(files), files)
    # file = files[0]
    random_id = np.random.randint(len(files))
    random_id = 22
    print("random_id", random_id)
    file = files[random_id]
    # for file in files:
    filename = path + "/" + file
    # print('filename', filename)
    # raise NotImplementedError
    # 读取harpoon的结果，返回为一个pandas dataframe格式数据，以及一个encoding结果
    with open (filename, 'rb') as f:
        dataframe, encoding_list = pickle.load(f)
        cur_dataframe = dataframe

    # 处理pandas表格数据
    header_row_num = len(dataframe.columns[0])
    # processed_filename = 'web/BackEnd/public/RL/pandas_' + file + ".xlsx"
    processed_filename = 'public/RL/pandas_' + file + ".xlsx"
    dataframe.to_excel(processed_filename) # 过渡阶段 没写好新的数据结构 暂时这么实现（把pandas表格转化为xlsx再用原本的函数读取）
    s = parse_sheet(processed_filename, header_row_num )
    tabular_content = s.result()   # 处理好的表格数据

    # 处理encoding结果
    encoding_res = []
    attributes = [attr for attr in dir(encoding_list[0]) if not attr.startswith("__")] # encoding list包含的所有属性名
    for item in encoding_list:
        obj = create_encoding_obj(item, attributes)
        encoding_res.append(obj)
    
    rl_obj = {}
    rl_obj["filename"] = file
    # rl_obj["table"] = str(tabular_content)
    rl_obj["content"] = tabular_content
    rl_obj["encoding"] = encoding_res
    
    json_rl_obj = json.dumps(rl_obj, cls=NpEncoder)
    refresh_dataset_list.append(json_rl_obj)

    # refresh_dataset_list的构成：[jsonstr, jsonstr, jsonstr...]
    os.remove(processed_filename)
    # os.remove(filename)#herehhhhhhhhhh
    # os.rename(filename, filename.replace('eval_results', 'eval_results_trash'))
    
    refresh_dataset_list = json.dumps(refresh_dataset_list)
    return refresh_dataset_list, dataframe

def get_alternative_dataset(data_name):
    if data_name == 'Console':
        print('alternative_dataset_list_Console', len(alternative_dataset_list_Console))
        return alternative_dataset_list_Console
    elif data_name == 'Income':
        print('alternative_dataset_list_Income', len(alternative_dataset_list_Income))
        return alternative_dataset_list_Income
    raise NotImplementedError

import copy
def load_alternative_dataset(data_name):
    if not isinstance(data_name, str):
        print(data_name)
        raise NotImplementedError
    if data_name == '保费收入分析.xlsx':
        data_name = 'Income'
    elif data_name == 'Console Sales.xlsx':
        data_name = 'Console'
    # print('data_name', data_name)
    # raise NotImplementedError
    path = 'eval_results_trash/' + data_name
    # raise NotImplementedError
    os.makedirs(path, exist_ok=True)
    files = os.listdir(path)
    if len(files) == 0:
        return []
    
    
    alternative_dataset_list = []
    files = os.listdir(path)
    # file = files[0]
    file = files[np.random.randint(len(files))]
    for file in files:
        filename = path + "/" + file
        # print('filename', filename)
        # 读取harpoon的结果，返回为一个pandas dataframe格式数据，以及一个encoding结果
        with open (filename, 'rb') as f:
            dataframe, encoding_list = pickle.load(f)

        # 处理pandas表格数据
        header_row_num = len(dataframe.columns[0])
        # processed_filename = 'web/BackEnd/public/RL/pandas_' + file + ".xlsx"
        processed_filename = 'public/RL/pandas_' + file + ".xlsx"
        # print('processed_filename', processed_filename)
        dataframe.to_excel(processed_filename) # 过渡阶段 没写好新的数据结构 暂时这么实现（把pandas表格转化为xlsx再用原本的函数读取）
        s = parse_sheet(processed_filename, header_row_num )
        tabular_content = s.result()   # 处理好的表格数据

        # 处理encoding结果
        encoding_res = []
        attributes = [attr for attr in dir(encoding_list[0]) if not attr.startswith("__")] # encoding list包含的所有属性名
        for item in encoding_list:
            obj = create_encoding_obj(item, attributes)
            encoding_res.append(obj)
        
        rl_obj = {}
        rl_obj["filename"] = file
        # rl_obj["table"] = str(tabular_content)
        rl_obj["content"] = tabular_content
        rl_obj["encoding"] = encoding_res
        
        json_rl_obj = json.dumps(rl_obj, cls=NpEncoder)
        alternative_dataset_list.append(json_rl_obj)

        # alternative_dataset_list的构成：[jsonstr, jsonstr, jsonstr...]
        os.remove(processed_filename)
        # os.remove(filename)#herehhhhhhhhhh
        # os.rename(filename, filename.replace('eval_results', 'eval_results_trash'))
    
    # print("alternative_dataset_list", len(alternative_dataset_list))
    alternative_dataset_list = json.dumps(alternative_dataset_list)
    if data_name == 'Console':
        global alternative_dataset_list_Console
        alternative_dataset_list_Console = copy.deepcopy(alternative_dataset_list)
    elif data_name == 'Income':
        global alternative_dataset_list_Income
        alternative_dataset_list_Income = copy.deepcopy(alternative_dataset_list)

