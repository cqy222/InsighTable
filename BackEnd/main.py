from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from dataset.tabular_data_collection import load_rl_dataset,load_alternative_dataset, get_alternative_dataset, load_tabular_dataset, get_tabular_dataset, get_rl_dataset, get_dataframe_list, get_refresh_dataset, get_dataframe, parse_upload_data
from ppo_table import update_next_encoding_dataset
from processing.tabular_data_parse import parse_sheet


import os, random
import json
import pandas as pd

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADER'] = 'Content-Type'

@app.route('/tabulardata', methods=['GET'])
@cross_origin()
def getTabularData():
    tabular_dataset = get_tabular_dataset()
    # print(tabular_dataset)
    # tabular_name_list = request.args.get('tabularData[]')
    return tabular_dataset

@app.route('/rldata', methods=['GET'])
@cross_origin()
def getRLData():
    rl_dataset = get_rl_dataset()
    return rl_dataset


# @app.route('/dflist', methods=['GET'])
# @cross_origin()
# def getDFList():
#     dataframe_list = get_dataframe_list()
#     return dataframe_list


@app.route('/uploadtabulardata', methods=['GET'])
@cross_origin()
def getUploadTabularData():
    file_name = request.args.get("name")
    data = parse_upload_data(file_name)
    # print("parse_data", data)
    path = "public/upload_" + file_name
    os.remove(path)  # 删除本地的文件
    return data

# @app.route('/recommendedConfig', methods=['GET'])
# @cross_origin()
# def getRecommendedConfigData():
#     data = parse_upload_data(file_name)
#     # print("parse_data", data)
#     path = "public/upload_" + file_name
#     os.remove(path)  # 删除本地的文件
#     return data


@app.route('/getupload', methods=['POST'])
@cross_origin()
def file_upload():
    # requ_data = {
    #     'file': request.files.get('file'),
    #     # 'file_info': dict(request.form)
    # }
    # resp_data = resp_file_upload(requ_data)
    
    requ_data = request.files.get('file')
    # suffix = requ_data.filename.split(".")[-1]
    # file_path = 'public/upload_temp.' + suffix
    file_name = requ_data.filename
    file_path = "public/upload_" + file_name
    requ_data.save(file_path)
    return "upload ok"

from ppo_table import Agent
from ppo_table import Table
from ppo_table import GCN
from ppo_table import GraphConvolution
@app.route('/refreshdata', methods=['GET'])
@cross_origin()
def file_refresh():
    file_name = request.args.get("name")
    rl_dataset, dataframe =  get_refresh_dataset(file_name)
    return jsonify([rl_dataset, dataframe.to_json()])


from visEncoding import VisEncoding
from processing.json_encoder import NpEncoder
from dataset.tabular_data_collection import create_encoding_obj

@app.route('/update_next_encoding', methods=['POST'])
@cross_origin()
def update_next_encoding_post():
    print('update_next_encoding_post')
    insight_list = request.json.get('insight_list')
    # selectedDF = request.json.get('selectedDF')
    selectedDF = get_dataframe()
    # print("selectedDF", selectedDF)
    # print("postinsight_list", insight_list)
    # print("postselectedDF", selectedDF)


    # print('request.selectedDF', type(selectedDF), selectedDF)
    # pdselectedDF = pd.DataFrame(selectedDF)
    # print('pdselectedDF', type(pdselectedDF), pdselectedDF)
    # print('pdselectedDFindex', pdselectedDF.index, pdselectedDF.columns)
    # raise NotImplementedError
    next_encoding_list = update_next_encoding_dataset(insight_list, selectedDF)
    print('next_encoding_list', next_encoding_list)

    if len(next_encoding_list) > 0:
        # 处理encoding结果
        encoding_res = []
        attributes = [attr for attr in dir(next_encoding_list[0]) if not attr.startswith("__")] # encoding list包含的所有属性名
        for item in next_encoding_list:
            obj = create_encoding_obj(item, attributes)
            encoding_res.append(obj)
        rl_obj = {}
        rl_obj["filename"] = "test_next_encoding"
        # rl_obj["content"] = tabular_content
        rl_obj["encoding"] = encoding_res
        
        json_rl_obj = json.dumps(rl_obj, cls=NpEncoder)
        
        print("json_rl_obj", json_rl_obj)
        # return next_encoding_list
        # next_encoding_list = asdict(next_encoding_list)
    else:
        json_rl_obj = []
    return json_rl_obj
    # def serialize_vis_encoding(obj):
    #     return obj.__dict__
    # return jsonify(next_encoding_list, default=serialize_vis_encoding)
    # return jsonify({'next_encoding_list': next_encoding_list})


# NotImplementedError
@app.route('/alternative_data', methods=['GET'])
@cross_origin()
def file_alternative():
    file_name = request.args.get("name")
    rl_dataset = get_alternative_dataset(file_name)
    return rl_dataset



# def resp_file_upload(requ_data):
#     # 保存文件
#     file_content = requ_data['file']
#     file_name = requ_data['file'].filename
#     file_path = 'public/' + file_name
#     if os.path.exists(file_path):
#         return { 'msg': '该文件已存在'}
#     else:
#     	file_content.save(file_path)
#     	return { 'msg': '保存文件成功' }


# def get_pandas(name): # 临时作为测试的函数
#     df = pd.DataFrame(pd.read_excel(name+'.elsx'))
#     return df

# @app.route('/pandas', methods=['GET'])
# @cross_origin()
# def handle_pandas():
#     # 现在为了写通pandas渲染，暂时采用后端等待用户输入序号的逻辑，最后应该改成阻塞等待harpoon计算出结果，再执行类似下边的逻辑
#     name = input("输入要导入的pandas序号：")
#     df = get_pandas(name) # 最后应该替换成获取harpoon生成pandas的函数，目前只是简单作为例子
#     data = load_pandas_data(df, name)
#     return data


if __name__ == "__main__":
    print('run 0.0.0.0:14450')
    load_tabular_dataset()
    load_rl_dataset()
    load_alternative_dataset('Console')
    load_alternative_dataset('Income')
    app.run(host='0.0.0.0', port=14450)





