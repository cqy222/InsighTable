import os
import copy
import datetime
import torch
import numpy as np
import pandas as pd
import pickle
# from gym import spaces
from rlpyt.spaces.int_box import IntBox
from rlpyt.spaces.float_box import FloatBox
from visEncoding import VisEncoding
# from .feature_extraction.extract import encode_tabular
from collections import deque
import operator
from scipy.spatial import distance
from scipy.optimize import curve_fit
from scipy.stats import entropy, norm, normaltest, mode, kurtosis, skew, skewtest, pearsonr, moment, linregress, kstest, chi2, f_oneway, alexandergovern, uniform
# from collections import namedtuple
from rlpyt.utils.collections import namedarraytuple
import warnings
from rlpyt.models.TUTA.utils import UNZIPS
EnvInfo = namedarraytuple("EnvInfo", ["game_score", "traj_done", "insight_ratio", "area_ratio", "Evenness_index"])
EnvSpaces = namedarraytuple("EnvSpaces", ["observation", "action"])
def power_law(x, alpha, beta):
    return alpha * np.power(x, beta)

class EvaENV(object):
    def __init__(self, tuta_tools):
        # self.


        # print(tuta_tools)
        self.args_tuta = tuta_tools[0]
        self.wiki = tuta_tools[1]
        self.tokenizer = tuta_tools[2]
        self.tuta = tuta_tools[3]
        # self.tuta = copy.deepcopy(tuta_tools[3])
        # raise NotImplementedError
        self.data_name = 'mytest_1'# Console Income TransT
        self.use_gnn = True
        self.use_tow_stage = True
        ratio = 0 #0.04
        self.hhhh = set()

        self.Jain = False
        self.log_enc = False
        self.end_ER = True
        self.end_add = 1
        self.addER = 4
        self.hand_ER = False
        self.Rcd = True
        self.new_ER = True
        self._insight_type_single = ['Outliers', 'Skewness', 'Kurtosis', 'Trend']
        self._insight_type_multiple = ['Pearsonr', 'M-Dominance', 'M-Top 2', 'M-Evenness']
        self._transformation_step = int(20 * ratio)
        self._insight_step = (20 - self._transformation_step) // len(self._insight_type_single + self._insight_type_multiple)
        if not self.use_tow_stage:
            self._transformation_step = 0
        if self.data_name == 'Console':
            # self.original_table = pd.read_excel('./rlpyt/envs/htable/Console Sales.xlsx', header=[0, 1], index_col=[0, 1, 2]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/Console Sales.csv', header=[0, 1, 2, 3], index_col=[0]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/Console Sales.csv', header=[0], index_col=[0, 1, 2, 3]).sort_index(axis=0).sort_index(axis=1)
            self.original_table = pd.read_csv('./rlpyt/envs/htable/Console Sales_less0_v2_out.csv', header=[0, 1, 2, 3], index_col=[0]).sort_index(axis=0).sort_index(axis=1)
        elif self.data_name == 'Income':
            # self.original_table = pd.read_excel('./rlpyt/envs/htable/保费收入分析.xlsx', header=[0, 1, 2], index_col=[0, 1, 2])
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析big.csv', header=[0], index_col=[0, 1, 2, 3, 4]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析.csv', header=[0], index_col=[0, 1, 2, 3, 4]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析_0319_1.csv', header=[0, 1, 2], index_col=[0, 1, 2]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析_0319_2.csv', header=[0, 1, 2, 3], index_col=[0, 1]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析_0319_3.csv', header=[0, 1, 2, 3], index_col=[0, 1]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析_0325.csv', header=[0], index_col=[0, 1, 2, 3, 4]).sort_index(axis=0).sort_index(axis=1)
            self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析_200423_8省_Trans.csv', header=[0, 1, 2, 3, 4], index_col=[0]).sort_index(axis=0).sort_index(axis=1)
            # self.original_table = pd.read_csv('./rlpyt/envs/htable/保费收入分析_200423_16省_Trans.csv', header=[0, 1, 2, 3, 4], index_col=[0]).sort_index(axis=0).sort_index(axis=1)
            self._must_select_index = [['保费合计', '同比增速'], ['当年', '当日', '当月']]
        elif self.data_name == 'GDP':
            self.original_table = pd.read_csv('./rlpyt/envs/htable/GDP.csv', header=[0], index_col=[0, 1]).sort_index(axis=0).sort_index(axis=1)
            self._must_select_index = []
        elif self.data_name == 'US':
            self.original_table = pd.read_csv('./rlpyt/envs/htable/US Investment Abroad.csv', header=[0, 1, 2], index_col=[0]).sort_index(axis=0).sort_index(axis=1)
            self._must_select_index = []
        # elif self.data_name == 'TransT':
        #     self.original_table = pd.read_csv('./rlpyt/envs/htable/trans_test_out.csv', header=[0, 1], index_col=[0, 1, 2, 3]).sort_index(axis=0).sort_index(axis=1)
        #     self._must_select_index = [['保费合计', '同比增速'], ['当年', '当日', '当月']]
        #     self._transformation_step = 50
        elif self.data_name == 'mytest_1':
            self.original_table = pd.read_csv('./rlpyt/envs/htable/mytest_1.csv', header=[0, 1], index_col=[0, 1]).sort_index(axis=0).sort_index(axis=1)
            self._must_select_index = []

            np.random.seed(0)
            self.random_seeds = np.random.randint(0, 1000000, 10000)
            

        self.use_rec = True
        if not self.use_gnn:
            ttt = self.original_table
            while isinstance(ttt, pd.DataFrame):
                ttt = ttt.stack()
            self.max_index_len = ttt.shape[0] * (len(ttt.index.names)+1)*2

        # self.original_table = pd.read_excel('./env/mytable.xlsx', header=[0, 1], index_col=[0, 1, 2]).fillna(method = 'ffill').sort_index()
        # self.original_table = pd.read_excel('./env/US Investment Abroad.xlsx', header=[0, 1], index_col=[0, 1]).fillna(method = 'ffill').sort_index(axis=0).sort_index(axis=1)
        # self.original_table = pd.read_excel('./env/School Curriculums.xlsx', header=[0, 1], index_col=[0, 1]).fillna(method = 'ffill').sort_index()
        # self.original_table = pd.read_excel('./env/GDP.xlsx', header=[0], index_col=[0, 1]).fillna(method = 'ffill').sort_index(axis=0).sort_index(axis=1)
        # self.original_table = pd.read_csv('./env/mytest.csv', header=[0, 1, 2], index_col=[0, 1]).sort_index(axis=0).sort_index(axis=1)


        self._insight_num = len(self._insight_type_single) + len(self._insight_type_multiple)
        self.episode_step = self._transformation_step + self._insight_num * self._insight_step - 1
        # self.original_table = pd.read_excel('./env/人力分析报表.xlsx', header=[0, 1, 2], index_col=[0])
        # self.episode_step = 600
        # print('original_table', self.original_table)
        # print('original_table', self.original_table.index)
        # print('original_table', self.original_table.columns)
        # raise NotImplementedError
        # self.original_table = self.original_table.unstack()
        # t1 = self.original_table.index.astype(pd.core.indexes.multi.MultiIndex)
        # t2 = self.original_table.columns
        # print(type(t1), t1)
        # print(type(t2), t2)
        self.row = len(self.original_table.index.names)
        self.col = len(self.original_table.columns.names)
        self.index_len = self.row + self.col - 1
        self.C_index_2 = (self.index_len) * (self.index_len - 1) // 2
        self.move_action_size = 3 + self.C_index_2 * 2
        self.vis_select_action_size = 8
        # self.action_size = 1 + self.move_action_size + self.vis_select_action_size + 1#first 1 -> no_op    last 1 -> vis
        self.action_size = 1 + self.move_action_size + self.vis_select_action_size#first 1 -> no_op
        self.action_space = IntBox(low=0, high=self.action_size)
        self.reset()
        # self.observation_space = FloatBox(low=-1, high=10, shape=self.obs_size)
        self.observation_space = self.obs_size
        self.spaces = EnvSpaces(
            observation=self.observation_space,
            action=self.action_space,
        )

        self.map_dir = {}
        cnt = 0
        for i in range(self.index_len):
            for j in range(i+1, self.index_len):
                self.map_dir[cnt] = [i, j]
                cnt += 1
        self.single_zone_max_area = 35
        # self.action_space = [spaces.Discrete(2), spaces.Discrete(6), spaces.Discrete(self.original_table.shape[1]), spaces.Box(low=-1, high=1, shape=[self.original_table.shape[1]])]
    
    def reset(self):
        self.table = copy.deepcopy(self.original_table)
        np.random.seed()
        np.random.seed(self.random_seeds[np.random.randint(10000)])
        nums = np.random.randn(self.table.shape[0], self.table.shape[1]) * 100000
        self.table.values[:] = nums

        self.mask = copy.deepcopy(self.original_table)
        self.mask_ins = copy.deepcopy(self.original_table).astype(int)
        orders = np.arange(self.table.shape[0]*self.table.shape[1], dtype=np.int64).reshape(*self.table.shape)
        self.mask_ins.values[:] = orders
        self.mask *= 0.

        self.bool_insight = np.zeros(self._insight_num + 2, dtype=np.bool)
        self.num_insight = np.zeros(self._insight_num + 2, dtype=np.int64)
        
        self.enc_list = []
        if self.use_gnn:
            headers = set(('root', 'left', 'top'))
            for i in range(len(self.table.index.names)):
                for j in self.table.index.get_level_values(i):
                    headers.add(j)
            for i in range(len(self.table.columns.names)):
                for j in self.table.columns.get_level_values(i):
                    headers.add(j)
            # headers = sorted(headers)
            self.H_set = headers
            np.random.seed(0)
            self.embeddings = np.random.random([len(self.H_set), len(self.H_set)])#replece with the stence emb in saved txt
            self.H_map = {}#for edge index
            for i, header in enumerate(headers):
                self.H_map[header] = i
        else:
            headers = set()
            for i in range(len(self.table.index.names)):
                for j in self.table.index.get_level_values(i):
                    headers.add(j)
            for i in range(len(self.table.columns.names)):
                for j in self.table.columns.get_level_values(i):
                    headers.add(j)
            # headers = sorted(headers)
            self.H_set = headers
            self.H_map = {}#for index id
            for i, header in enumerate(headers):
                self.H_map[header] = (1+i) / len(self.H_set)
            # self.index_map = {}#顺序无关，因为这是features
            # cnt = 1
            # for i in range(len(self.table.index.names)):
            #     for j in self.table.index.get_level_values(i):
            #         self.index_map[j] = cnt / (1 + self.index_len)
            #     cnt += 1
            # for i in range(len(self.table.columns.names)):
            #     for j in self.table.columns.get_level_values(i):
            #         self.index_map[j] = cnt / (1 + self.index_len)
            #     cnt += 1
        # for (k,v) in self.index_map.items():
        #     print('===', k, v)
        if self.use_tow_stage:
            self.stage = 0
            self.stage0_size = 1 + self.move_action_size
        self.select_row = self.table.index[0]
        self.select_col = self.table.columns[0]
        # self.select_row = self.table.index[self.table.shape[0]//2]
        # self.select_col = self.table.columns[self.table.shape[1]//2]
        self.row = len(self.table.index.names)
        self.col = len(self.table.columns.names)
        if self.row == 1:
            self.select_row = (self.select_row,)
        if self.col == 1:
            self.select_col = (self.select_col,)
        # print('self.select_row', self.select_row, type(self.select_row))
        # print('self.select_col', self.col, self.select_col, type(self.select_col))
        # raise NotImplementedError
        self.steps = 0
        obs = self.get_obs()
        if self.use_gnn:
            # self.obs_size = obs[2].shape
            self.obs_size = (obs[0].shape, obs[2].shape)
        else:
            self.obs_size = obs.shape
        self.episodeR = 0.

        self.sb_trans_cnt = 0
        # self.old_diversity = 0
        self.old_ER = 0
        return obs#TODO cpu优化 TODO multiindex input TODO mask input

    def tuta_enc(self, table_json):
        # import time
        # st = time.time()
        dataset = self.wiki.worker(0, [table_json])[0]
        # print('dataset', type(dataset), len(dataset), dataset)
        # raise NotImplementedError

        token_matrix, number_matrix, position_lists, header_info, format_or_text = dataset
        # print('token_matrix', type(token_matrix), len(token_matrix))
        # print('number_matrix', type(number_matrix), len(number_matrix))
        # print('position_lists', type(position_lists), len(position_lists))
        # print('header_info', type(header_info), len(header_info))
        # print('format_or_text', type(format_or_text), len(format_or_text))
        # raise NotImplementedError
        format_matrix, context = None, None
        if isinstance(format_or_text, str):    # wiki, title
            context = (format_or_text, )
        elif isinstance(format_or_text, list): # sheet, format_matrix
            format_matrix = format_or_text
        elif isinstance(format_or_text, tuple): # wdc, context = (title, page_title, text_before, text_after)
            context = format_or_text
        else:
            print("Unsupported data type at last position: ", type(format_or_text))
        sampling_matrix = self.tokenizer.sampling(
            token_matrix=token_matrix, 
            number_matrix=number_matrix, 
            header_info=header_info, 
            max_disturb_num=self.args_tuta.max_disturb_num, 
            disturb_prob=self.args_tuta.disturb_prob, 
            clc_rate=self.args_tuta.clc_rate
        )
        # print('sampling_matrix', len(sampling_matrix), len(sampling_matrix[0]), (np.array(sampling_matrix)).sum(), sampling_matrix)
        # raise NotImplementedError
        results = self.tokenizer.objective_preprocess(
            sampling_matrix=sampling_matrix, 
            token_matrix=token_matrix, 
            number_matrix=number_matrix, 
            position_lists=position_lists, 
            format_matrix=format_matrix, 
            context=context, 
            add_sep=self.args_tuta.add_separate
        )
        # print('results', len(results), len(results[0]), results[0])
        if results is None:
            raise NotImplementedError
        if len(results[0]) > self.args_tuta.max_cell_num:
            raise NotImplementedError
        token_seq = [tok for cell in results[0] for tok in cell]
        # print('token_seq', len(token_seq), token_seq)
        # if len(token_seq) > self.args_tuta.max_seq_len:
        #     return None
        # raise NotImplementedError
        tok_list, num_list, pos_list, fmt_list, cell_ind, cell_mlm, cell_clc, cell_tcr = results #token_list, num_list, pos_list, format_list, indicator, mlm_label, clc_label, tcr_label 
        token_id, num_mag, num_pre, num_top, num_low = [], [], [], [], []
        token_order, pos_row, pos_col, pos_top, pos_left = [], [], [], [], []
        format_vec, indicator = [], []
        mlm_label, clc_label, tcr_label = [], [], []
        cell_num = len(tok_list)
        for icell in range(cell_num):
            tokens = tok_list[icell]
            cell_len = len(tokens)
            token_id.extend(tokens)
            token_order.extend([ii for ii in range(cell_len)])
            mlm_label.extend(cell_mlm[icell])

            num_feats = num_list[icell]
            num_mag.extend([f[0] for f in num_feats])
            num_pre.extend([f[1] for f in num_feats])
            num_top.extend([f[2] for f in num_feats])
            num_low.extend([f[3] for f in num_feats])

            row, col, ttop, tleft = pos_list[icell]
            pos_row.extend([row for _ in range(cell_len)])
            pos_col.extend([col for _ in range(cell_len)])
            entire_top = UNZIPS['tuta'](ttop, self.args_tuta.node_degree, sum(self.args_tuta.node_degree))
            pos_top.extend([entire_top for _ in range(cell_len)])
            entire_left = UNZIPS['tuta'](tleft, self.args_tuta.node_degree, sum(self.args_tuta.node_degree))
            pos_left.extend([entire_left for _ in range(cell_len)])

            format_vec.extend( [fmt_list[icell] for _ in range(cell_len)] )
            indicator.extend(cell_ind[icell])
            clc_label.extend(cell_clc[icell])
            tcr_label.extend(cell_tcr[icell])
        # print('token', time.time() - st)
        # st = time.time()

        # for i in range(0, len(token_id), 125):#LongTensor #IntTensor
        # if self.rank > 10:
        #     device = torch.device("cuda:1")
        # else:
        #     device = torch.device("cuda:0")
        
        # device = torch.device("cuda:0")
        # token_id = torch.IntTensor(token_id).unsqueeze(0).cuda(device=device, non_blocking=True)
        # num_mag = torch.IntTensor(num_mag).unsqueeze(0).cuda(device=device, non_blocking=True)
        # num_pre = torch.IntTensor(num_pre).unsqueeze(0).cuda(device=device, non_blocking=True)
        # num_top = torch.IntTensor(num_top).unsqueeze(0).cuda(device=device, non_blocking=True)
        # num_low = torch.IntTensor(num_low).unsqueeze(0).cuda(device=device, non_blocking=True)
        # token_order = torch.IntTensor(token_order).unsqueeze(0).cuda(device=device, non_blocking=True)
        # pos_row = torch.IntTensor(pos_row).unsqueeze(0).cuda(device=device, non_blocking=True)
        # pos_col = torch.IntTensor(pos_col).unsqueeze(0).cuda(device=device, non_blocking=True)
        # pos_top = torch.LongTensor(pos_top).unsqueeze(0).cuda(device=device, non_blocking=True)
        # pos_left = torch.LongTensor(pos_left).unsqueeze(0).cuda(device=device, non_blocking=True)
        # format_vec = torch.FloatTensor(format_vec).unsqueeze(0).cuda(device=device, non_blocking=True)
        # indicator = torch.IntTensor(indicator).unsqueeze(0).cuda(device=device, non_blocking=True)
        # mlm_label = torch.IntTensor(mlm_label).unsqueeze(0).cuda(device=device, non_blocking=True)
        # clc_label = torch.IntTensor(clc_label).unsqueeze(0).cuda(device=device, non_blocking=True)
        # tcr_label = torch.IntTensor(tcr_label).unsqueeze(0).cuda(device=device, non_blocking=True)
        # print('333', time.time() - st)
        # st = time.time()
        # print('--------------------')
        # print('tuta', self.tuta)
        # print('token_id', token_id.shape, token_id.device, token_id.requires_grad)
        # print('num_mag', num_mag.shape)
        # print('num_pre', num_pre.shape)
        # print('num_top', num_top.shape)
        # print('num_low', num_low.shape)
        # print('token_order', token_order.shape)
        # print('pos_row', pos_row.shape)
        # print('pos_col', pos_col.shape)
        # print('pos_top', pos_top.shape)
        # print('pos_left', pos_left.shape)
        # print('format_vec', format_vec.shape)
        # print('indicator', indicator.shape)
        # print('tuta', next(self.tuta.parameters()).device)
        # print('tuta', next(self.tuta.parameters()).requires_grad)
        # raise NotImplementedError

        device = self.tuta.parameters().__next__().device
        token_id = torch.IntTensor(token_id).unsqueeze(0).to(device=device, non_blocking=True)
        num_mag = torch.IntTensor(num_mag).unsqueeze(0).to(device=device, non_blocking=True)
        num_pre = torch.IntTensor(num_pre).unsqueeze(0).to(device=device, non_blocking=True)
        num_top = torch.IntTensor(num_top).unsqueeze(0).to(device=device, non_blocking=True)
        num_low = torch.IntTensor(num_low).unsqueeze(0).to(device=device, non_blocking=True)
        token_order = torch.IntTensor(token_order).unsqueeze(0).to(device=device, non_blocking=True)
        pos_row = torch.IntTensor(pos_row).unsqueeze(0).to(device=device, non_blocking=True)
        pos_col = torch.IntTensor(pos_col).unsqueeze(0).to(device=device, non_blocking=True)
        pos_top = torch.LongTensor(pos_top).unsqueeze(0).to(device=device, non_blocking=True)
        pos_left = torch.LongTensor(pos_left).unsqueeze(0).to(device=device, non_blocking=True)
        format_vec = torch.FloatTensor(format_vec).unsqueeze(0).to(device=device, non_blocking=True)
        indicator = torch.IntTensor(indicator).unsqueeze(0).to(device=device, non_blocking=True)
        mlm_label = torch.IntTensor(mlm_label).unsqueeze(0).to(device=device, non_blocking=True)
        clc_label = torch.IntTensor(clc_label).unsqueeze(0).to(device=device, non_blocking=True)
        tcr_label = torch.IntTensor(tcr_label).unsqueeze(0).to(device=device, non_blocking=True)
        # print('*'*50, device)
        # raise NotImplementedError

        # print('tensor', time.time() - st)
        # st = time.time()

        embs = self.tuta(
            token_id, num_mag, num_pre, num_top, num_low, 
            token_order, pos_row, pos_col, pos_top, pos_left, format_vec, indicator
        )
        embs = embs.squeeze(0).mean(0)
        if device.type != 'cpu':
            embs = embs.cpu()
        # print('embs', time.time() - st)
        # raise NotImplementedError
        return embs
  
    def get_obs(self):
        warnings.filterwarnings("ignore")
        if self.use_tow_stage:
            # print('=' * 10, self.steps, self._transformation_step)
            if self.steps == self._transformation_step:
                self.stage = 1
                self.select_row = self.table.index[0]
                self.select_col = self.table.columns[0]
                self.row = len(self.table.index.names)
                self.col = len(self.table.columns.names)
                # self.select_row = self.table.index[self.table.shape[0]//2]
                # self.select_col = self.table.columns[self.table.shape[1]//2]
                if self.row == 1:
                    self.select_row = (self.select_row,)
                if self.col == 1:
                    self.select_col = (self.select_col,)
            if self.steps >= self._transformation_step:
                # print('self.steps - self._transformation_step', (self.steps - self._transformation_step) % self._insight_step)
                # if (self.steps - self._transformation_step) % self._insight_step == 0:
                #     self.select_row = self.table.index[self.table.shape[0]//2]
                #     self.select_col = self.table.columns[self.table.shape[1]//2]
                #     if self.row == 1:
                #         self.select_row = (self.select_row,)
                #     if self.col == 1:
                #         self.select_col = (self.select_col,)
                self.cur_insight_type = (self.steps - self._transformation_step) // self._insight_step
            else:
                self.cur_insight_type = -1
            if self.use_gnn:
                edge_index = np.zeros([len(self.H_set), len(self.H_set)], dtype=np.bool)
                edge_index[self.H_map['root']][self.H_map['left']] = True
                edge_index[self.H_map['root']][self.H_map['top']] = True
                for i in self.table.index:
                    if isinstance(i, tuple):
                        edge_index[self.H_map['left'], self.H_map[i[0]]] = True
                        for j in range(1, len(i)):
                            edge_index[self.H_map[i[j-1]], self.H_map[i[j]]] = True
                    else:
                        edge_index[self.H_map['left'], self.H_map[i]] = True
                for i in self.table.columns:
                    if isinstance(i, tuple):
                        edge_index[self.H_map['top'], self.H_map[i[0]]] = True
                        for j in range(1, len(i)):
                            edge_index[self.H_map[i[j-1]], self.H_map[i[j]]] = True
                    else:
                        edge_index[self.H_map['top'], self.H_map[i]] = True
                edge_index += edge_index.T
                # np.random.seed(None)
                # edge_index = np.random.randint(0,2,(len(self.H_set), len(self.H_set))) #^^^^^

                # np.set_printoptions(threshold=np.inf)
                # for (k,v) in self.H_map.items():
                #     print('===', k, v)
                # print('table', self.table.index[0], self.table.columns[0])
                # print('select', self.select_row, self.select_col)
                # print('edge_index', edge_index.shape, edge_index)
                # x = np.zeros([len(self.H_set), 2])
                # for i in range(len(self.H_set)):
                #     x[i][1] = (i+1)*1./len(self.H_set)
                # if self.stage == 1:
                #     for index in self.select_row:
                #         if index in self.H_set:
                #             x[self.H_map[index]][0] = -1
                #     for index in self.select_col:
                #         if index in self.H_set:
                #             x[self.H_map[index]][0] = 1
                x = np.zeros([len(self.H_set), len(self.H_set)*3])
                x[:,0:len(self.H_set)] = self.embeddings
                if self.stage == 1:
                    x[:,len(self.H_set):len(self.H_set)*2] = -(self.cur_insight_type+1)/self._insight_num
                    x[:,len(self.H_set)*2:] = -1
                    for index in self.select_row:
                        if index in self.H_set:
                            x[self.H_map[index]][len(self.H_set)*2:] = 2
                    for index in self.select_col:
                        if index in self.H_set:
                            x[self.H_map[index]][len(self.H_set)*2:] = 1
                # np.random.seed(None)
                # x[:,:len(self.H_set)] = np.random.randint(-1,2,(len(self.H_set), len(self.H_set)))
                # print('x', x.shape)
                # raise NotImplementedError
                

                table_datas = self.table.values.flatten()
                table_datas = (table_datas - table_datas.mean()) / table_datas.std()
                if self.mask_ins.isnull().values.any():
                    z = np.zeros([self.original_table.shape[0]*self.original_table.shape[1]])
                else:
                    masks = self.mask.values.flatten() *1./self._insight_num
                    z = np.zeros_like(masks)
                    z[self.mask_ins.values.flatten()] = masks
                    # try:
                    #     z[self.mask_ins.values.flatten()] = masks
                    # except Exception as ex:
                    #     print('self.mask_ins', self.mask_ins.shape, self.mask_ins)
                    #     print('masks', masks.shape, masks)
                    #     raise NotImplementedError
                datas = np.concatenate((np.array([self.stage, self.stage0_size/100., (self.cur_insight_type+1)/self._insight_num]), z))
                obs = (x, edge_index, datas)
            else:#not use_gnn
                # # -------------------- older version --------------------
                # # order1 = np.zeros([self.table.shape[0], len(self.table.index.names)])
                # # if self.stage == 1:
                # #     if self.row == 1:
                # #         if len(self.select_row) == 1:
                # #             p1 = self.table.index.get_loc(self.select_row[0])
                # #         else:
                # #             p1 = 0
                # #     else:
                # #         p1 = self.table.index.get_loc(self.select_row)
                # #     if isinstance(p1, slice):
                # #         p1 = p1.start
                # #     if self.col == 1:
                # #         if len(self.select_col) == 1:
                # #             p2 = self.table.columns.get_loc(self.select_col[0])
                # #         else:
                # #             p2 = 0
                # #     else:
                # #         p2 = self.table.columns.get_loc(self.select_col)
                # #     if isinstance(p2, slice):
                # #         p2 = p2.start


                # # for i, index in enumerate(self.table.index):
                # #     if len(self.table.index.names) > 1:
                # #         for j, name in enumerate(index):
                # #             if self.stage == 1 and i == p1 and name in self.select_row:
                # #                 order1[i][j] = -self.H_map[name]
                # #             else:
                # #                 order1[i][j] = self.H_map[name]
                # #     else:
                # #         if self.stage == 1 and i == p1 and index in self.select_row:
                # #             order1[i][0] = -self.H_map[index]
                # #         else:
                # #             order1[i][0] = self.H_map[index]
                # # order2 = np.zeros([self.table.shape[1], len(self.table.columns.names)])
                # # for i, col in enumerate(self.table.columns):
                # #     if len(self.table.columns.names) > 1:
                # #         for j, name in enumerate(col):
                # #             if self.stage == 1 and i == p2 and name in self.select_col:
                # #                 order2[i][j] = -self.H_map[name]
                # #             else:
                # #                 order2[i][j] = self.H_map[name]
                # #     else:
                # #         if self.stage == 1 and i == p2 and col in self.select_col:
                # #             order2[i][0] = -self.H_map[col]
                # #         else:
                # #             order2[i][0] = self.H_map[col]
                # # order = np.concatenate((order1.flatten(), order2.flatten()))
                # # order = np.concatenate((np.zeros(self.max_index_len - order.shape[0]), order))
                # # # self.hhhh.add(hash(order.tostring()))
                # # # print(order.shape, order.max(), order.min(), order.mean(), order.std(), hash(order.tostring()), len(self.hhhh))
                # # table_datas = self.table.values.flatten()
                # # table_datas = (table_datas - table_datas.mean()) / table_datas.std()
                # # if self.mask_ins.isnull().values.any():
                # #     z = np.zeros([self.original_table.shape[0]*self.original_table.shape[1]])
                # # else:
                # #     masks = self.mask.values.flatten() *1./self._insight_num
                # #     z = np.zeros_like(masks)
                # #     z[self.mask_ins.values.flatten()] = masks
                # # obs = np.concatenate((np.array([self.stage, self.stage0_size/100., self.cur_insight_type]), order, z))

                # arr = self.mask.values / self._insight_num
                # if np.isnan(arr).any():
                #     arr = np.zeros_like(arr)
                # col_np = None
                # for i in range(len(self.table.columns.names)):
                #     col_i = copy.deepcopy(self.table.columns.get_level_values(i).to_numpy())
                #     if col_np is None:
                #         col_np = col_i
                #     else:
                #         col_np = np.vstack((col_np, col_i))
                # col_np = np.vstack((col_np, arr))
                # row_np = None
                # for i in range(len(self.table.index.names)):
                #     row_i = copy.deepcopy(self.table.index.get_level_values(i).to_numpy())
                #     if row_np is None:
                #         row_np = row_i
                #     else:
                #         row_np = np.vstack((row_np, row_i))
                # if len(row_np.shape) == 1:
                #     row_np = np.expand_dims(row_np, 0)
                # row_np = np.hstack((np.zeros([len(self.table.index.names), len(self.table.columns.names)]), row_np))
                # z = np.hstack((row_np.T, col_np)).flatten()
                # for i in self.H_set:
                #     if i in self.select_row or i in self.select_col:
                #         z[z == i] = - self.H_map[i]
                #     else:
                #         z[z == i] = self.H_map[i]
                # z = z.astype(np.float32)
                # if np.isnan(z).any():
                #     np.set_printoptions(threshold=np.inf)
                #     print(z)
                #     raise NotImplementedError
                # # self.hhhh.add(hash(z.tostring()))
                # # print(z.shape, z.max(), z.min(), z.mean(), z.std(), hash(z.tostring()), len(self.hhhh), self.select_row, self.select_col)
                # z = np.concatenate((z, np.zeros(self.max_index_len - z.shape[0])))
                # obs = np.concatenate((np.array([self.stage, self.stage0_size/100., self.cur_insight_type]), z))
                # -------------------- older version --------------------

                table = copy.deepcopy(self.table)
                table[abs(self.mask)>1e-7] = -100000000000 # TODO fix
                table_json = self.pandas2json(table)
                # print('table', table)
                # print('table_json', table_json)
                # raise NotImplementedError

                with torch.no_grad():
                    table_emb = self.tuta_enc(table_json)
                if(table_emb == None):
                    raise NotImplementedError
                table_emb = table_emb.numpy()
                # print('table_emb', type(table_emb), table_emb.shape, table_emb)
                # raise NotImplementedError
                obs = np.concatenate((np.array([self.stage, self.stage0_size/100., self.cur_insight_type]), table_emb))
        else:#not two stage
            self.cur_insight_type = self.steps // self._insight_step
            if self.use_gnn:
                edge_index = np.zeros([len(self.H_set), len(self.H_set)], dtype=np.bool)
                edge_index[self.H_map['root']][self.H_map['left']] = True
                edge_index[self.H_map['root']][self.H_map['top']] = True
                for i in self.table.index:
                    if isinstance(i, tuple):
                        edge_index[self.H_map['left'], self.H_map[i[0]]] = True
                        for j in range(1, len(i)):
                            edge_index[self.H_map[i[j-1]], self.H_map[i[j]]] = True
                    else:
                        edge_index[self.H_map['left'], self.H_map[i]] = True
                for i in self.table.columns:
                    if isinstance(i, tuple):
                        edge_index[self.H_map['top'], self.H_map[i[0]]] = True
                        for j in range(1, len(i)):
                            edge_index[self.H_map[i[j-1]], self.H_map[i[j]]] = True
                    else:
                        edge_index[self.H_map['top'], self.H_map[i]] = True
                edge_index += edge_index.T

                x = np.zeros([len(self.H_set), len(self.H_set)*3])
                x[:,0:len(self.H_set)] = self.embeddings
                x[:,len(self.H_set):len(self.H_set)*2] = -(self.cur_insight_type+1)/self._insight_num
                x[:,len(self.H_set)*2:] = -1

                table_datas = self.table.values.flatten()
                table_datas = (table_datas - table_datas.mean()) / table_datas.std()
                if self.mask_ins.isnull().values.any():
                    z = np.zeros([self.original_table.shape[0]*self.original_table.shape[1]])
                else:
                    masks = self.mask.values.flatten() *1./self._insight_num
                    z = np.zeros_like(masks)
                    z[self.mask_ins.values.flatten()] = masks
                datas = np.concatenate((np.array([(self.cur_insight_type+1)/self._insight_num]), z))
                obs = (x, edge_index, datas)
            else:#not use_gnn
                # order1 = np.zeros([self.table.shape[0], len(self.table.index.names)])
                # if self.row == 1:
                #     if len(self.select_row) == 1:
                #         p1 = self.table.index.get_loc(self.select_row[0])
                #     else:
                #         p1 = 0
                # else:
                #     p1 = self.table.index.get_loc(self.select_row)
                # if isinstance(p1, slice):
                #     p1 = p1.start
                # if self.col == 1:
                #     if len(self.select_col) == 1:
                #         p2 = self.table.columns.get_loc(self.select_col[0])
                #     else:
                #         p2 = 0
                # else:
                #     p2 = self.table.columns.get_loc(self.select_col)
                # if isinstance(p2, slice):
                #     p2 = p2.start
                # for i, index in enumerate(self.table.index):
                #     if len(self.table.index.names) > 1:
                #         for j, name in enumerate(index):
                #             if i == p1 and name in self.select_row:
                #                 order1[i][j] = -self.H_map[name]
                #             else:
                #                 order1[i][j] = self.H_map[name]
                #     else:
                #         if i == p1 and index in self.select_row:
                #             order1[i][0] = -self.H_map[index]
                #         else:
                #             order1[i][0] = self.H_map[index]
                # order2 = np.zeros([self.table.shape[1], len(self.table.columns.names)])
                # for i, col in enumerate(self.table.columns):
                #     if len(self.table.columns.names) > 1:
                #         for j, name in enumerate(col):
                #             if i == p2 and name in self.select_col:
                #                 order2[i][j] = -self.H_map[name]
                #             else:
                #                 order2[i][j] = self.H_map[name]
                #     else:
                #         if i == p2 and col in self.select_col:
                #             order2[i][0] = -self.H_map[col]
                #         else:
                #             order2[i][0] = self.H_map[col]
                # order = np.concatenate((order1.flatten(), order2.flatten()))
                # order = np.concatenate((np.zeros(self.max_index_len - order.shape[0]), order))
                # table_datas = self.table.values.flatten()
                # table_datas = (table_datas - table_datas.mean()) / table_datas.std()
                # if self.mask_ins.isnull().values.any():
                #     z = np.zeros([self.original_table.shape[0]*self.original_table.shape[1]])
                # else:
                #     masks = self.mask.values.flatten() *1./self._insight_num
                #     z = np.zeros_like(masks)
                #     z[self.mask_ins.values.flatten()] = masks
                # obs = np.concatenate((np.array([self.cur_insight_type]), order, z))
                
                arr = self.mask.values / self._insight_num
                if np.isnan(arr).any():
                    arr = np.zeros_like(arr)
                col_np = None
                for i in range(len(self.table.columns.names)):
                    col_i = copy.deepcopy(self.table.columns.get_level_values(i).to_numpy())
                    if col_np is None:
                        col_np = col_i
                    else:
                        col_np = np.vstack((col_np, col_i))
                col_np = np.vstack((col_np, arr))
                row_np = None
                for i in range(len(self.table.index.names)):
                    row_i = copy.deepcopy(self.table.index.get_level_values(i).to_numpy())
                    if row_np is None:
                        row_np = row_i
                    else:
                        row_np = np.vstack((row_np, row_i))
                if len(row_np.shape) == 1:
                    row_np = np.expand_dims(row_np, 0)
                row_np = np.hstack((np.zeros([len(self.table.index.names), len(self.table.columns.names)]), row_np))
                z = np.hstack((row_np.T, col_np)).flatten()
                for i in self.H_set:
                    if i in self.select_row or i in self.select_col:
                        z[z == i] = - self.H_map[i]
                    else:
                        z[z == i] = self.H_map[i]
                z = z.astype(np.float32)
                if np.isnan(z).any():
                    np.set_printoptions(threshold=np.inf)
                    print(z)
                    raise NotImplementedError
                # self.hhhh.add(hash(z.tostring()))
                # print(z.shape, z.max(), z.min(), z.mean(), z.std(), hash(z.tostring()), len(self.hhhh), self.select_row, self.select_col)
                z = np.concatenate((z, np.zeros(self.max_index_len - z.shape[0])))
                obs = np.concatenate((np.array([self.cur_insight_type]), z))
        return obs
    
    def my_loc(self, df, select_row, select_col):
        if select_row == () and select_col == ():
            return df
        elif select_row == () and select_col != ():
            return df.loc[:, select_col]
        elif select_row != () and select_col == ():
            return df.loc[select_row, :]
        else:
            return df.loc[select_row, select_col]
    
    def my_change(self, df, select_row, select_col, value):
        # if not isinstance(select_row, tuple) or not isinstance(select_col, tuple):
        #     print('select_row', select_row, type(select_row))
        #     print('select_col', select_col, type(select_col))
        #     raise NotImplementedError
        pre_sum = df.values.sum()
        if select_row == () and select_col == ():
            df = df * 0 + value
        elif select_row == () and select_col != ():
            df.loc[:, select_col] = value
        elif select_row != () and select_col == ():
            df.loc[select_row, :] = value
        else:
            df.loc[select_row, select_col] = value
        if pre_sum == df.values.sum():
            raise NotImplementedError
        return df
    
    def jude_must_select_index(self):
        if hasattr(self, '_must_select_index'):
            for must_select_index in self._must_select_index:
                bool_flag = False
                for j in must_select_index:
                    if j in self.select_row or j in self.select_col:
                        bool_flag = True
                        break
                if not bool_flag:
                    return False
        return True
    
    def func_Outstanding(self, v):
        if (v < 0).any() or (abs(v) < 1e-7).all():
            return False
        v = np.sort(v)[::-1]
        ins = range(1, v.size + 1)
        try:
            law_params, _ = curve_fit(power_law, ins[1:], v[1:], maxfev=5000)
        except Exception as ex:
            return False
        prediction = power_law(ins, *law_params)
        residuals = v - prediction
        loc, scale = norm.fit(residuals[1:])
        dist_params = {'loc': loc, 'scale': scale}
        p_value = norm.sf(residuals[0], **dist_params)
        return p_value < 0.05


    def func_Outliers(self, v):#1.5IQR 3IQR分级
        q1, q25, q75, q99 = np.percentile(v, [1, 25, 75, 99])
        iqr = q75 - q25
        outliers_15iqr = np.logical_or(v < (q25 - 1.5 * iqr), v > (q75 + 1.5 * iqr)).any()
        outliers_3iqr = np.logical_or(v < (q25 - 3 * iqr), v > (q75 + 3 * iqr)).any()
        # outliers_1_99 = np.logical_or(v < q1, v > q99).any()
        # outliers_3std = np.logical_or(v < (np.mean(v) - 3 * np.std(v)),
        #                                 v > (np.mean(v) + 3 * np.std(v))).any()
        return [outliers_15iqr, outliers_3iqr, self.func_Outstanding(v)]

    def func_Skewness(self, v):
        if (v>=0).all():
            v = v[abs(v) > 1e-7]
        if v.shape[0] >= 10:
            skewness = abs(skew(v))
            return skewness > 1.5
        return False
    
    def func_Kurtosis(self, v):
        if (v>=0).all():
            v = v[abs(v) > 1e-7]
        if v.shape[0] >= 10:
            kur = abs(kurtosis(v))
            return kur > 2
        return False

    def func_Trend(self, v):
        slope, intercept, r_value, p_value, std_err = linregress(np.arange(v.shape[0]), v)
        trend = r_value ** 2 * (1 - p_value)
        return trend > 0.7
    
    def func_Dominance(self, v):
        if (v<0).any() or (v<1e-7).sum() / v.shape[0] > 0.5 or v.shape[0] <= 1:
            return -1
        v = v / v.sum()
        return (v > 0.5).any()
    
    def func_Top_2(self, v):
        if (v<0).any() or (v<1e-7).sum() / v.shape[0] > 0.5 or v.shape[0] <= 1:
            return -1
        v = v / v.sum()
        return (v > 0.34).sum() == 2
    
    def func_Evenness(self, v):
        if (abs(v)<1e-7).sum() / v.shape[0] > 0.5 or v.shape[0] <= 1:
            return -1
        return abs(v.std() / v.mean()) < 0.05
    
    def func_Pearsonr(self, tmp):
        if tmp.shape[0] > 4:
            return False#chaos too many
        p_value = tmp.T.corr(method=lambda x, y: pearsonr(x, y)[1]).values
        # ttt = tmp.T.corr(method=lambda x, y: pearsonr(x, y)[0]).values
        # p_value[ttt < 0] = 1
        p_value[np.isnan(p_value)] = 1
        p_value = p_value < 0.05
        if p_value.max(-1).sum() >= p_value.shape[0] - 1:
            return True
        return False

    def step(self, action):
        action = int(action)
        # if self.log_enc:
        #     print('action', self.steps, action)
        # raise NotImplementedError
        # print('action', self.action_size, action)
        # print("select_row", self.select_row, self.select_col)
        reward = 0
        #TODO Tolinear
        done = False
        if self.use_tow_stage:
            if self.stage == 0 and action >= self.stage0_size or self.stage == 1 and 0 < action < self.stage0_size:
                print(self.steps, self.stage, action, self.stage0_size)
                raise NotImplementedError
        
        # action = 0#sb_selecttest
        # np.random.seed(None)
        # action = np.random.randint(1 + self.move_action_size + self.vis_select_action_size)
        if (np.abs(self.mask.values.round()) < 1e-5).sum() == 0:
            # done = True
            pass
        elif action == 0:#no op
            pass
        elif action < 1 + self.move_action_size:
            action -= 1
            # print('----mask', self.mask.values.sum())
            # np.random.seed(None)
            # action = np.random.randint(self.move_action_size)
            # if self.log_enc:
            #     print('action-trans', action)
            table = self.table
            mask = self.mask
            mask_ins = self.mask_ins
            #TODO stack不自动对index排序
            if action == 0:#transpose
                table = table.T
                mask = mask.T
                mask_ins = mask_ins.T
                if self.log_enc:
                    print('action-transpose', self.steps)
            elif action == 1:#stack
                if len(table.columns.names) > 1:
                    table = table.stack()
                    mask = mask.stack()
                    mask_ins = mask_ins.stack()
                    if self.log_enc:
                        print('action-stack', self.steps)
            elif action == 2:#unstack
                if len(table.index.names) > 1:
                    table = table.unstack()
                    mask = mask.unstack()
                    mask_ins = mask_ins.unstack()
                    if self.log_enc:
                        print('action-unstack', self.steps)
            elif action < 3 + self.C_index_2:#swaplevel
                action -= 3
                px = self.map_dir[action][0]
                py = self.map_dir[action][1]
                if px < len(table.index.names) and py < len(table.index.names):
                    table = table.swaplevel(px, py, axis=0)
                    mask = mask.swaplevel(px, py, axis=0)
                    mask_ins = mask_ins.swaplevel(px, py, axis=0)
                    if self.log_enc:
                        print('action-swaplevel', self.steps, px, py)
            else:
                action -= 3 + self.C_index_2
                px = self.map_dir[action][0]
                py = self.map_dir[action][1]
                if px < len(table.columns.names) and py < len(table.columns.names):
                    table = table.swaplevel(px, py, axis=1)
                    mask = mask.swaplevel(px, py, axis=1)
                    mask_ins = mask_ins.swaplevel(px, py, axis=1)
                    if self.log_enc:
                        print('action-swaplevel', self.steps, px, py)
            table_drop = table.dropna(0)
            if table_drop.shape[0]*table_drop.shape[1] == self.original_table.shape[0]*self.original_table.shape[1]:
                table = table_drop
                mask = mask.dropna(0)
                mask_ins = mask_ins.dropna(0).astype(int)
            else:
                table_drop = table.dropna(1)
                if table_drop.shape[0]*table_drop.shape[1] == self.original_table.shape[0]*self.original_table.shape[1]:
                    table = table_drop
                    mask = mask.dropna(1)
                    mask_ins = mask_ins.dropna(1).astype(int)
            
            self.table = table.sort_index(axis=0).sort_index(axis=1)
            self.mask = mask.sort_index(axis=0).sort_index(axis=1)
            self.mask_ins = mask_ins.sort_index(axis=0).sort_index(axis=1)
            self.sb_trans_cnt += 1
            # print('self.mask_ins', type(self.mask_ins))
         
                # print('======', self.table.index[0], self.table.columns[0])
            # else:
            #     reward = -0.1
            

            self.select_row = self.table.index[0]
            self.select_col = self.table.columns[0]
            # self.select_row = self.table.index[self.table.shape[0]//2]
            # self.select_col = self.table.columns[self.table.shape[1]//2]
            self.row = len(self.table.index.names)
            self.col = len(self.table.columns.names)
            if self.row == 1:
                self.select_row = (self.select_row,)
            if self.col == 1:
                self.select_col = (self.select_col,)
            # if (abs(self.mask.values)>1e-7).any():
            #     self.mask *= 0
            #     reward = -1
            # print('====mask', self.mask.values.sum())
            # print('table', self.table.shape, self.table.isnull().values.any())
        elif action < 1 + self.move_action_size + self.vis_select_action_size and not self.table.isnull().values.any():
            action -= 1 + self.move_action_size
            # action = 0###sb_selecttest
            # np.random.seed(None)
            # action = np.random.randint(0,self.vis_select_action_size)
            # if self.log_enc:
            #     print('action-select', action)
                
            #TODO by name index
            if action == 0:#select_row 减
                self.select_row = self.select_row[:-1]
                # if self.log_enc:
                #     print('action-select_row_less')
            elif action == 1:#select_col 减
                self.select_col = self.select_col[:-1]
                # if self.log_enc:
                #     print('action-select_col_less')
            elif action == 2:#select_row 加
                if self.row == 1:
                    if len(self.select_row) == 1:
                        pass
                    else:
                        self.select_row = (self.table.index[0],)
                else:
                    p = self.table.index.get_loc(self.select_row)
                    if isinstance(p, slice):
                        p = p.start
                    l = len(self.select_row)
                    self.select_row = self.table.index[p][:l+1]
                # if self.log_enc:
                #     print('action-select_row_more')
            elif action == 3:#select_col 加
                if self.col == 1:
                    if len(self.select_col) == 1:
                        pass
                    else:
                        self.select_col = (self.table.columns[0],)
                else:
                    p = self.table.columns.get_loc(self.select_col)
                    if isinstance(p, slice):
                        p = p.start
                    l = len(self.select_col)
                    self.select_col = self.table.columns[p][:l+1]
                # if self.log_enc:
                #     print('action-select_col_more')

            elif action == 4:#select_row 下移
                if self.row == 1:
                    if len(self.select_row) == 1:
                        p = self.table.index.get_loc(self.select_row[0])
                        p += 1
                        if p >= len(self.table.index):
                            p -= 1
                        self.select_row = (self.table.index[p],)
                    else:
                        pass
                else:
                    p = self.table.index.get_loc(self.select_row)
                    if isinstance(p, slice):
                        p = p.stop
                    else:
                        p += 1
                    if p >= len(self.table.index):
                        p -= 1
                    l = len(self.select_row)
                    self.select_row = self.table.index[p][:l]
                # if self.log_enc:
                #     print('action-select_row_down')
            elif action == 5:#select_col 下移
                if self.col == 1:
                    if len(self.select_col) == 1:
                        p = self.table.columns.get_loc(self.select_col[0])
                        p += 1
                        if p >= len(self.table.columns):
                            p -= 1
                        self.select_col = (self.table.columns[p],)
                    else:
                        pass
                else:
                    p = self.table.columns.get_loc(self.select_col)
                    if isinstance(p, slice):
                        p = p.stop
                    else:
                        p += 1
                    if p >= len(self.table.columns):
                        p -= 1
                    l = len(self.select_col)
                    self.select_col = self.table.columns[p][:l]
                # if self.log_enc:
                #     print('action-select_col_down')
            elif action == 6:#select_row 上移
                if self.row == 1:
                    if len(self.select_row) == 1:
                        p = self.table.index.get_loc(self.select_row[0])
                        p -= 1
                        if p < 0:
                            p = 0
                        self.select_row = (self.table.index[p],)
                    else:
                        pass
                else:
                    p = self.table.index.get_loc(self.select_row)
                    if isinstance(p, slice):
                        p = p.start
                    p -= 1
                    if p < 0:
                        p = 0
                    l = len(self.select_row)
                    self.select_row = copy.deepcopy(self.table.index[p][:l])
                # if self.log_enc:
                #     print('action-select_row_up')
            elif action == 7:#select_col 上移
                if self.col == 1:
                    if len(self.select_col) == 1:
                        p = self.table.columns.get_loc(self.select_col[0])
                        p -= 1
                        if p < 0:
                            p = 0
                        self.select_col = (self.table.columns[p],)
                    else:
                        pass
                else:
                    p = self.table.columns.get_loc(self.select_col)
                    if isinstance(p, slice):
                        p = p.start
                    p -= 1
                    if p < 0:
                        p = 0
                    if self.col == 1:
                        l = 1
                    else:
                        l = len(self.select_col)
                    self.select_col = copy.deepcopy(self.table.columns[p][:l])
                # if self.log_enc:
                #     print('action-select_col_up')
        
        # elif action < 1 + self.move_action_size + self.vis_select_action_size + 1:
            # if self.log_enc:
            #     print('action-vis', self.cur_insight_type)
            if self.cur_insight_type < len(self._insight_type_single):
                cur_insight_type = self._insight_type_single[self.cur_insight_type]
            else:
                cur_insight_type = self._insight_type_multiple[self.cur_insight_type-len(self._insight_type_single)]
            # print('cur_insight_type', self.cur_insight_type, cur_insight_type)
            # if 'select_row' not in locals().keys():
            #     self.select_row = self.table.index[0]
            #     self.select_col = self.table.columns[0]
            #     self.row = len(self.table.index.names)
            #     self.col = len(self.table.columns.names)
            #     if self.row == 1:
            #         self.select_row = (self.select_row,)
            #     if self.col == 1:
            #         self.select_col = (self.select_col,)

            tmp = self.my_loc(self.table, self.select_row, self.select_col)
            pre_shape = tmp.shape
            pre_tmp = copy.deepcopy(tmp)
            is_horizontal = None
            if len(self.select_row) == self.row or pre_shape[0] == 1:
                is_horizontal = True
            elif len(self.select_col) == self.col or len(pre_shape) > 1 and pre_shape[1] == 1:
                is_horizontal = False
            # if isinstance(tmp, pd.Series) or (len(pre_shape) > 0 and pre_shape[0] == 1) or (len(pre_shape) == 2 and pre_shape[1] == 1):
            #     print('pre_shape', pre_shape, self.select_row, self.select_col, is_horizontal)
            tmp_mask = self.my_loc(self.mask, self.select_row, self.select_col) 
            if not isinstance(tmp, pd.Series) and not isinstance(tmp, pd.DataFrame):
                pass
            elif len(self.select_row) == 0 or len(self.select_col) == 0:
                pass
            elif not self.jude_must_select_index():
                pass
            elif (np.abs(tmp_mask.values.round()) > 1e-5).sum() == 0 and 3 < tmp.values.flatten().shape[0] <= self.single_zone_max_area:
                #TODO p value 0.05 0.01分级
                #TODO 同时存在多种怎么办，选择哪一种
                v = tmp.values.flatten()

                bool_has = False
                bool_reced = False
                if cur_insight_type == 'Outliers':
                    Outliers = self.func_Outliers(v)
                    if sum(Outliers):
                        bool_has = True
                elif cur_insight_type == 'Skewness':
                    # if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1 and len(self.select_row) == self.row-1 and len(self.select_col) == self.col-1:
                    if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1:
                        bool_has = self.func_Skewness(v)
                elif cur_insight_type == 'Kurtosis':
                    # if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1 and len(self.select_row) == self.row-1 and len(self.select_col) == self.col-1:
                    if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1:
                        bool_has = self.func_Kurtosis(v)
                elif cur_insight_type == 'Trend':
                    if is_horizontal!=False and (isinstance(tmp, pd.DataFrame) and tmp.shape[0] == 1 and tmp.shape[1] >= 3 or isinstance(tmp, pd.Series) and tmp.shape[0] >= 3):
                        if isinstance(tmp, pd.Series):
                            ttt = tmp.index[0]
                        else:
                            ttt = tmp.columns[0]
                        if isinstance(ttt, tuple):
                            ttt = ttt[-1]
                        if isinstance(ttt, np.number):
                            bool_has = self.func_Trend(v)
                        # if bool_has:
                        #     if isinstance(tmp, pd.Series):
                        #         ttt = tmp.index[0]
                        #     else:
                        #         ttt = tmp.columns[0]
                        #     if isinstance(ttt, tuple):
                        #         ttt = ttt[-1]
                        #     print('DataFrame', type(ttt), isinstance(ttt, (int, float)), isinstance(ttt, np.number))
                        #     # raise NotImplementedError
                elif cur_insight_type == 'M-Dominance':
                    # if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1 and len(self.select_row) == self.row-1 and len(self.select_col) == self.col-1:
                    if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1:
                        Dominance = self.func_Dominance(tmp.values.sum(-1))
                        if Dominance == True:
                            bool_has = True
                            is_horizontal = False
                        else:
                            Dominance = self.func_Dominance(tmp.values.sum(0))
                            if Dominance == True:
                                bool_has = True
                                is_horizontal = True
                elif cur_insight_type == 'M-Top 2':
                    # if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1 and len(self.select_row) == self.row-1 and len(self.select_col) == self.col-1:
                    if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1:
                        Top_2 = self.func_Top_2(tmp.values.sum(-1))
                        if Top_2 == True:
                            bool_has = True
                            is_horizontal = False
                        else:
                            Top_2 = self.func_Top_2(tmp.values.sum(0))
                            if Top_2 == True:
                                bool_has = True
                                is_horizontal = True
                elif cur_insight_type == 'M-Evenness':
                    # if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1 and len(self.select_row) == self.row-1 and len(self.select_col) == self.col-1:
                    if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1:
                        Evenness = self.func_Evenness(tmp.values.sum(-1))
                        if Evenness == True:
                            bool_has = True
                            is_horizontal = False
                        else:
                            Evenness = self.func_Evenness(tmp.values.sum(0))
                            if Evenness == True:
                                bool_has = True
                                is_horizontal = True
                elif cur_insight_type == 'Pearsonr':
                    # if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1 and len(self.select_row) == self.row-1 and len(self.select_col) == self.col-1:
                    if isinstance(tmp, pd.DataFrame) and tmp.shape[0] > 1 and tmp.shape[1] > 1:
                        Pearsonr = self.func_Pearsonr(tmp)
                        Pearsonr_T = self.func_Pearsonr(tmp.T)
                        if Pearsonr:
                            is_horizontal = True
                            bool_has = True
                        elif Pearsonr_T:
                            is_horizontal = False
                            bool_has = True
                else:
                    raise NotImplementedError

                if self.use_rec:
                    if bool_has:
                        for i in range(len(self.select_row), -2, -1):
                            if i == -1:
                                bad_rec_row = False
                                recommend_row = [False for _ in range(self.table.shape[0])]
                                if len(self.select_row) >= 1:
                                    recommend_row = [False for _ in range(self.table.shape[0])]
                                    for k, index in enumerate(self.table.index):
                                        if isinstance(index, int):
                                            bad_rec_row = True
                                            break
                                        if self.select_row[-1] in index:
                                            recommend_row[k] = True
                                if bad_rec_row:
                                    continue
                            else:
                                recommend_row = self.select_row[:i]
                            for j in range(len(self.select_col), -2, -1):
                                if j == -1:
                                    bad_rec_col = False
                                    recommend_col = [False for _ in range(self.table.shape[1])]
                                    if len(self.select_col) >= 1:
                                        for k, col in enumerate(self.table.columns):
                                            if isinstance(col, int):
                                                bad_rec_col = True
                                                break
                                            if self.select_col[-1] in col:
                                                recommend_col[k] = True
                                    if bad_rec_col:
                                        continue
                                else:
                                    recommend_col = self.select_col[:j]
                                tmp = self.my_loc(self.table, recommend_row, recommend_col)
                                tmp_mask = self.my_loc(self.mask, recommend_row, recommend_col)
                                if not isinstance(tmp, pd.DataFrame):
                                    pass
                                elif (np.abs(tmp_mask.values.round()) > 1e-5).sum() > 0:
                                    if i != -1 and j != -1:
                                        break
                                elif tmp.shape[0] > 1 and tmp.shape[1] > 1 and 200 > tmp.values.flatten().shape[0] > v.shape[0]:

                                    # print('pre_tmp', pre_shape, pre_tmp)
                                    # print('tmp', tmp.shape, tmp, tmp.index, tmp.columns)
                                    # print('select_row', self.select_row, self.select_col, recommend_row, recommend_col)
                                    
                                    tmp_list = []#[data]
                                    rec_list = []#[(pos, bool)]
                                    cur_index = []
                                    cur_columns = []
                                    if isinstance(recommend_row, list):
                                        rec_row = self.table.index[recommend_row]
                                        for index in rec_row:
                                            ttt = index[:len(self.select_row)]
                                            if ttt not in cur_index:
                                                cur_index.append(ttt)
                                    else:
                                        for index in tmp.index:
                                            if not isinstance(index, tuple):
                                                index = (index,)
                                            ttt = recommend_row + index
                                            ttt = ttt[:len(self.select_row)]
                                            if ttt not in cur_index:
                                                cur_index.append(ttt)
                                    
                                    if isinstance(recommend_col, list):
                                        rec_col = self.table.columns[recommend_col]
                                        for columns in rec_col:
                                            ttt = columns[:len(self.select_col)]
                                            if ttt not in cur_columns:
                                                cur_columns.append(ttt)
                                    else:
                                        for columns in tmp.columns:
                                            if not isinstance(columns, tuple):
                                                columns = (columns,)
                                            ttt = recommend_col + columns
                                            ttt = ttt[:len(self.select_col)]
                                            if ttt not in cur_columns:
                                                cur_columns.append(ttt)

                                    bool_max_zone = False
                                    for index in cur_index:
                                        for columns in cur_columns:
                                            tmp_list.append(self.my_loc(self.table, index, columns))
                                            if tmp_list[-1].values.flatten().shape[0] > self.single_zone_max_area:
                                                bool_max_zone = True
                                                break

                                            if len(index) == 0:
                                                pos_row = slice(0, self.table.shape[0], None)
                                            elif self.row == 1:
                                                pos_row = self.table.index.get_loc(index[0])
                                            else:
                                                pos_row = self.table.index.get_loc(index)
                                            if len(columns) == 0:
                                                pos_col = slice(0, self.table.shape[1], None)
                                            elif self.col == 1:
                                                pos_col = self.table.columns.get_loc(columns[0])
                                            else:
                                                pos_col = self.table.columns.get_loc(columns)
                                            rec_list.append((pos_row, pos_col))
                                        if bool_max_zone:
                                            break
                                    if bool_max_zone:
                                        continue

                                    # if cur_insight_type not in self._insight_type_multiple:
                                    #     try:
                                    #         pre_rec_shape = tmp.shape#all block reshape之前的shape
                                    #         tmp = tmp.values.reshape(-1, v.shape[0])
                                    #     except Exception as ex:
                                    #         continue
       
                                    # if cur_insight_type in ['M-Dominance', 'M-Top 2', 'M-Evenness']:
                                    #     try:
                                    #         pre_rec_shape = tmp.shape
                                    #         if is_horizontal == True:
                                    #             tmp = tmp.values.reshape(-1, pre_shape[0]).sum(-1)
                                    #             tmp = tmp.reshape(-1, pre_shape[1])
                                    #         elif is_horizontal == False:
                                    #             tmp = tmp.values.reshape(-1, pre_shape[1]).sum(-1)
                                    #             tmp = tmp.reshape(-1, pre_shape[0])
                                    #     except Exception as ex:
                                    #         continue
                                    if cur_insight_type in self._insight_type_single:
                                        tmp_list = [tmp.values.flatten() for tmp in tmp_list]
                                    if cur_insight_type in ['M-Dominance', 'M-Top 2', 'M-Evenness']:
                                        if is_horizontal == True:
                                            tmp_list = [tmp.values.sum(0) for tmp in tmp_list]
                                        elif is_horizontal == False:
                                            tmp_list = [tmp.values.sum(-1) for tmp in tmp_list]
                                        else:
                                            raise NotImplementedError
                                    if cur_insight_type == 'Pearsonr':
                                        cur_tol_area = np.array([tmp.values.flatten().shape[0] for tmp in tmp_list]).sum()
                                    else:
                                        cur_tol_area = np.array([tmp.flatten().shape[0] for tmp in tmp_list]).sum()


                                    if cur_insight_type == 'Outliers':
                                        Outliers_s = map(self.func_Outliers, tmp_list)
                                        Outliers_s = np.array(list(Outliers_s))
                                        if len(pre_shape) == 2 and pre_shape[0] > 1 and pre_shape[1] > 1:
                                            Outliers_tmp = Outliers_s.max(-1)
                                            # if Outliers_tmp.sum() >= Outliers_tmp.shape[0] - 1 and Outliers_tmp.sum() > 1:
                                            if Outliers_tmp.sum() >= Outliers_tmp.shape[0] - 1:
                                                bool_reced = True
                                                Outliers_vis_type = 'unit visualization'
                                        else:
                                            Outliers_tmp = Outliers_s[:,0]
                                            # if Outliers_tmp.sum() >= Outliers_tmp.shape[0] - 1 and Outliers_tmp.sum() > 1:
                                            if Outliers_tmp.sum() >= Outliers_tmp.shape[0] - 1:
                                                bool_reced = True
                                                Outliers_vis_type = 'box plot'
                                            else:
                                                Outliers_tmp = Outliers_s[:,2]
                                                # if Outliers_tmp.sum() >= Outliers_tmp.shape[0] - 1 and Outliers_tmp.sum() > 1:
                                                if Outliers_tmp.sum() >= Outliers_tmp.shape[0] - 1:
                                                    bool_reced = True
                                                    Outliers_vis_type = 'bar chart'
                                        rec_insight_flag = Outliers_tmp
                                    elif cur_insight_type == 'Skewness':
                                        Skewness_s = map(self.func_Skewness, tmp_list)
                                        Skewness_s = np.array(list(Skewness_s))
                                        rec_insight_flag = Skewness_s
                                        # if Skewness_s.sum() == Skewness_s.shape[0] - 1 and Skewness_s.sum() > 1:#== not >=
                                        if Skewness_s.sum() == Skewness_s.shape[0] - 1:#== not >=
                                            bool_reced = True
                                    elif cur_insight_type == 'Kurtosis':
                                        Kurtosis_s = map(self.func_Kurtosis, tmp_list)
                                        Kurtosis_s = np.array(list(Kurtosis_s))
                                        rec_insight_flag = Kurtosis_s
                                        # if Kurtosis_s.sum() == Kurtosis_s.shape[0] - 1 and Kurtosis_s.sum() > 1:#== not >=
                                        if Kurtosis_s.sum() == Kurtosis_s.shape[0] - 1:#== not >=
                                            bool_reced = True
                                    elif cur_insight_type == 'Trend':
                                        Trend_s = map(self.func_Trend, tmp_list)
                                        Trend_s = np.array(list(Trend_s))
                                        rec_insight_flag = Trend_s
                                        # if Trend_s.sum() >= Trend_s.shape[0] - 1 and Trend_s.sum() > 1:
                                        if Trend_s.sum() >= Trend_s.shape[0] - 1:
                                            bool_reced = True
                                    elif cur_insight_type == 'M-Dominance':
                                        Dominance_s = map(self.func_Dominance, tmp_list)
                                        Dominance_s = list(Dominance_s)
                                        if -1 not in Dominance_s:
                                            Dominance_s = np.array(Dominance_s)
                                            rec_insight_flag = Dominance_s
                                            # if Dominance_s.sum() >= Dominance_s.shape[0] - 1 and Dominance_s.sum() > 1:
                                            if Dominance_s.sum() >= Dominance_s.shape[0] - 1:
                                                bool_reced = True
                                    elif cur_insight_type == 'M-Top 2':
                                        Top_2_s = map(self.func_Top_2, tmp_list)
                                        Top_2_s = list(Top_2_s)
                                        if -1 not in Top_2_s:
                                            Top_2_s = np.array(Top_2_s)
                                            rec_insight_flag = Top_2_s
                                            # if Top_2_s.sum() >= Top_2_s.shape[0] - 1 and Top_2_s.sum() > 1:
                                            if Top_2_s.sum() >= Top_2_s.shape[0] - 1:
                                                bool_reced = True
                                    elif cur_insight_type == 'M-Evenness':
                                        Evenness_s = map(self.func_Evenness, tmp_list)
                                        Evenness_s = list(Evenness_s)
                                        if -1 not in Evenness_s:
                                            Evenness_s = np.array(Evenness_s)
                                            rec_insight_flag = Evenness_s
                                            # if Evenness_s.sum() >= Evenness_s.shape[0] - 1 and Evenness_s.sum() > 1:
                                            if Evenness_s.sum() >= Evenness_s.shape[0] - 1:
                                                bool_reced = True
                                    elif cur_insight_type == 'Pearsonr':
                                        if is_horizontal == True:
                                            Pearsonr_s = map(self.func_Pearsonr, tmp_list)
                                            Pearsonr_s = np.array(list(Pearsonr_s))
                                            rec_insight_flag = Pearsonr_s
                                            # if Pearsonr_s.sum() >= Pearsonr_s.shape[0] - 1 and Pearsonr_s.sum() > 1:
                                            if Pearsonr_s.sum() >= Pearsonr_s.shape[0] - 1:
                                                bool_reced = True
                                            # print('Pearsonr_s', Pearsonr_s.shape, Pearsonr_s)
                                            # raise NotImplementedError
                                            
                                        elif is_horizontal == False:
                                            Pearsonr_s = map(self.func_Pearsonr, [tmp.T for tmp in tmp_list])
                                            Pearsonr_s = np.array(list(Pearsonr_s))
                                            rec_insight_flag = Pearsonr_s
                                            # if Pearsonr_s.sum() >= Pearsonr_s.shape[0] - 1 and Pearsonr_s.sum() > 1:
                                            if Pearsonr_s.sum() >= Pearsonr_s.shape[0] - 1:
                                                bool_reced = True
                                            # print('Pearsonr_s', Pearsonr_s.shape, Pearsonr_s)
                                            # raise NotImplementedError
                                        else:
                                            raise NotImplementedError
                                    else:
                                        raise NotImplementedError



                                    if bool_reced:
                                        if(rec_insight_flag > 1).any():
                                            print('rec_insight_flag', rec_insight_flag, cur_insight_type)
                                            raise NotImplementedError
                                        if cur_insight_type == 'Outliers':
                                            if Outliers_vis_type == 'unit visualization':
                                                self.bool_insight[len(self.bool_insight)-2] = True
                                                cur_num_insight = self.num_insight[len(self.bool_insight)-2]
                                                if self.new_ER:
                                                    self.num_insight[len(self.bool_insight)-2] += cur_tol_area
                                                else:
                                                    self.num_insight[len(self.bool_insight)-2] += rec_insight_flag.sum()
                                                self.mask = self.my_change(self.mask, recommend_row, recommend_col, -(len(self.bool_insight)-1))
                                                self.mask = self.my_change(self.mask, self.select_row, self.select_col, len(self.bool_insight)-1)
                                            elif Outliers_vis_type == 'bar chart':
                                                self.bool_insight[len(self.bool_insight)-1] = True
                                                cur_num_insight = self.num_insight[len(self.bool_insight)-1]
                                                if self.new_ER:
                                                    self.num_insight[len(self.bool_insight)-1] += cur_tol_area
                                                else:
                                                    self.num_insight[len(self.bool_insight)-1] += rec_insight_flag.sum()
                                                self.mask = self.my_change(self.mask, recommend_row, recommend_col, -len(self.bool_insight))
                                                self.mask = self.my_change(self.mask, self.select_row, self.select_col, len(self.bool_insight))
                                            else:
                                                self.bool_insight[self.cur_insight_type] = True
                                                cur_num_insight = self.num_insight[self.cur_insight_type]
                                                if self.new_ER:
                                                    self.num_insight[self.cur_insight_type] += cur_tol_area
                                                else:
                                                    self.num_insight[self.cur_insight_type] += rec_insight_flag.sum()
                                                self.mask = self.my_change(self.mask, recommend_row, recommend_col, -(self.cur_insight_type+1))
                                                self.mask = self.my_change(self.mask, self.select_row, self.select_col, self.cur_insight_type+1)
                                        else:
                                            self.bool_insight[self.cur_insight_type] = True
                                            cur_num_insight = self.num_insight[self.cur_insight_type]
                                            if self.new_ER:
                                                self.num_insight[self.cur_insight_type] += cur_tol_area
                                            else:
                                                self.num_insight[self.cur_insight_type] += rec_insight_flag.sum()
                                            self.mask = self.my_change(self.mask, recommend_row, recommend_col, -(self.cur_insight_type+1))
                                            self.mask = self.my_change(self.mask, self.select_row, self.select_col, self.cur_insight_type+1)
                                        if self.hand_ER:
                                            if rec_insight_flag.sum() == rec_insight_flag.shape[0] - 1:
                                                cur_tol_area *= 2
                                        else:
                                            if rec_insight_flag.sum() == rec_insight_flag.shape[0] - 1:
                                                cur_tol_area *= 2
                                            else:
                                                cur_tol_area *= 1.5
                                        if abs(cur_tol_area) < 1e-7:
                                            raise NotImplementedError
                                        # if cur_insight_type == 'Skewness' or cur_insight_type == 'Kurtosis':
                                        #     cur_tol_area /= 4
                                        if self.Rcd:
                                            insight_value = cur_tol_area
                                        else:
                                            if cur_insight_type != 'Outliers':
                                                insight_value = rec_insight_flag.sum()*2#9
                                            else:
                                                insight_value = rec_insight_flag.sum()/10*2#3
                                        # if cur_insight_type == 'Pearsonr':
                                        #     insight_value = 2 * 3 * np.clip(10./tmp.shape[1], 0, 1)
                                        # else:
                                        #     insight_value = 2 * np.clip(10./v.shape[0], 0, 1)
                                        if self.hand_ER:
                                            reward += 15/(cur_num_insight+20) * insight_value
                                            # print('15/(cur_num_insight+20)', 15/(cur_num_insight+20), insight_value, reward)
                                        else:
                                            reward += insight_value
                                        if self.log_enc:
                                            if len(self.select_row) == 0:
                                                pos_row = slice(0, self.table.shape[0], None)
                                            elif self.row == 1:
                                                pos_row = self.table.index.get_loc(self.select_row[0])
                                            else:
                                                pos_row = self.table.index.get_loc(self.select_row)
                                            if len(self.select_col) == 0:
                                                pos_col = slice(0, self.table.shape[1], None)
                                            elif self.col == 1:
                                                pos_col = self.table.columns.get_loc(self.select_col[0])
                                            else:
                                                pos_col = self.table.columns.get_loc(self.select_col)

                                            rec_all_list = []
                                            for rec, flag in zip(rec_list, rec_insight_flag):
                                                rec_all_list.append((rec, flag))
                                            rec_list = rec_all_list
                                                
                                            if i == -1:
                                                rec_row_type = 'name'
                                                rec_row_priority = 1
                                            else:
                                                rec_row_type = 'subtree'
                                                rec_row_priority = len(self.select_row) - i
                                            if j == -1:
                                                rec_col_type = 'name'
                                                rec_col_priority = 1
                                            else:
                                                rec_col_type = 'subtree'
                                                rec_col_priority = len(self.select_col) - j
                                            if cur_insight_type == 'Outliers':
                                                if Outliers_vis_type == 'unit visualization':
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='unit visualization', insight_type='Outliers', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                elif Outliers_vis_type == 'box plot':
                                                    if is_horizontal == True:
                                                        vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='box plot', x='value', y='column '+str(self.row), color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value, \
                                                                            rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                    elif is_horizontal == False:
                                                        vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='box plot', x='row '+str(self.col), y='value', color='row '+str(self.col), is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value, \
                                                                            rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                    else:
                                                        raise NotImplementedError
                                                elif Outliers_vis_type == 'bar chart':
                                                    if is_horizontal == True:
                                                        vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='row '+str(self.col), y='value', is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value, \
                                                                            rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                    elif is_horizontal == False:
                                                        vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='value', y='column '+str(self.row), is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value, \
                                                                            rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                    else:
                                                        raise NotImplementedError
                                                else:
                                                    print('pre_shape', pre_shape, tmp.shape)
                                                    raise NotImplementedError()
                                            elif cur_insight_type == 'Skewness':
                                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='histogram area', x='value', y='density', insight_type='Skewness', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                            elif cur_insight_type == 'Kurtosis':
                                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='histogram bar', x='value', y='density', insight_type='Kurtosis', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                            elif cur_insight_type == 'Trend':
                                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='horizon graph', x='row '+str(self.col), y='value', is_horizontal=True, insight_type='Trend', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                            elif cur_insight_type == 'M-Dominance':
                                                if is_horizontal == True:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='row '+str(self.col), theta='value', radius='value', insight_type='M-Dominance', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                elif is_horizontal == False:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='column '+str(self.row), theta='value', radius='value', insight_type='M-Dominance', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                else:
                                                    raise NotImplementedError
                                            elif cur_insight_type == 'M-Top 2':
                                                if is_horizontal == True:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='row '+str(self.col), theta='value', radius='value', insight_type='M-Dominance', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                elif is_horizontal == False:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='column '+str(self.row), theta='value', radius='value', insight_type='M-Dominance', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                else:
                                                    raise NotImplementedError
                                            elif cur_insight_type == 'M-Evenness':
                                                if is_horizontal == True:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='row '+str(self.col), y='value', color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='M-Evenness', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                elif is_horizontal == False:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='value', y='column '+str(self.row), color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='M-Evenness', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                else:
                                                    raise NotImplementedError
                                            elif cur_insight_type == 'Pearsonr':
                                                if is_horizontal == True:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='multi line chart', x='row '+str(self.col), y='value', color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='Pearsonr', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                elif is_horizontal == False:
                                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='multi line chart', x='value', y='column '+str(self.row), color='row '+str(self.col), is_horizontal=is_horizontal, insight_type='Pearsonr', insight_value=insight_value, \
                                                                        rec_row_type=rec_row_type, rec_row_priority=rec_row_priority, rec_col_type=rec_col_type, rec_col_priority=rec_col_priority, rec_list=rec_list)
                                                else:
                                                    raise NotImplementedError
                                            if vis_info != None:
                                                self.enc_list.append(vis_info)
                                        break
                            
                            if (np.abs(tmp_mask.values.round()) > 1e-5).sum() > 0 and i != -1 and j != -1 or bool_reced:
                                break



                if bool_has and not bool_reced and cur_insight_type != 'Skewness':
                    cur_tol_area = pre_shape[0]*pre_shape[1] if len(pre_shape)==2 else pre_shape[0]
                    if abs(cur_tol_area) < 1e-7:
                        raise NotImplementedError
                    if cur_insight_type == 'Outliers':
                        if len(pre_shape) == 2 and pre_shape[0] > 1 and pre_shape[1] > 1:
                            self.bool_insight[len(self.bool_insight)-2] = True
                            cur_num_insight = self.num_insight[len(self.bool_insight)-2]
                            if self.new_ER:
                                self.num_insight[len(self.bool_insight)-2] += cur_tol_area
                            else:
                                self.num_insight[len(self.bool_insight)-2] += 1
                            self.mask = self.my_change(self.mask, self.select_row, self.select_col, len(self.bool_insight)-1)
                        elif Outliers[2]:
                            self.bool_insight[len(self.bool_insight)-1] = True
                            cur_num_insight = self.num_insight[len(self.bool_insight)-1]
                            if self.new_ER:
                                self.num_insight[len(self.bool_insight)-1] += cur_tol_area
                            else:
                                self.num_insight[len(self.bool_insight)-1] += 1
                            self.mask = self.my_change(self.mask, self.select_row, self.select_col, len(self.bool_insight))
                        else:
                            self.bool_insight[self.cur_insight_type] = True
                            cur_num_insight = self.num_insight[self.cur_insight_type]
                            if self.new_ER:
                                self.num_insight[self.cur_insight_type] += cur_tol_area
                            else:
                                self.num_insight[self.cur_insight_type] += 1
                            self.mask = self.my_change(self.mask, self.select_row, self.select_col, self.cur_insight_type+1)
                    else:
                        self.bool_insight[self.cur_insight_type] = True
                        cur_num_insight = self.num_insight[self.cur_insight_type]
                        if self.new_ER:
                            self.num_insight[self.cur_insight_type] += cur_tol_area
                        else:
                            self.num_insight[self.cur_insight_type] += 1
                        self.mask = self.my_change(self.mask, self.select_row, self.select_col, self.cur_insight_type+1)
                    # insight_value = 1 + 1. / v.shape[0]
                    # if cur_insight_type == 'Kurtosis':
                    #     cur_tol_area /= 4
                    if self.Rcd:
                        insight_value = cur_tol_area
                    else:
                        if cur_insight_type != 'Outliers':
                            insight_value = 1
                        else:
                            insight_value = 0.1
                    if self.hand_ER:
                        reward += 15/(cur_num_insight+20) * insight_value
                    else:
                        reward += insight_value
                    if self.log_enc:
                        # print('self.table.index', self.table.index, self.select_row)
                        # print('self.table.columns', self.table.columns, self.select_col)
                        if len(self.select_row) == 0:
                            pos_row = slice(0, self.table.shape[0], None)
                        elif self.row == 1:
                            pos_row = self.table.index.get_loc(self.select_row[0])
                        else:
                            pos_row = self.table.index.get_loc(self.select_row)
                        if len(self.select_col) == 0:
                            pos_col = slice(0, self.table.shape[1], None)
                        elif self.col == 1:
                            pos_col = self.table.columns.get_loc(self.select_col[0])
                        else:
                            pos_col = self.table.columns.get_loc(self.select_col)
                        # print('pos_row', pos_row, self.select_row)
                        # print('pos_col', pos_col, self.select_col)
                        if cur_insight_type == 'Outliers':
                            if len(pre_shape) == 2 and pre_shape[0] > 1 and pre_shape[1] > 1:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='unit visualization', insight_type='Outliers', insight_value=insight_value)
                            elif Outliers[2]:
                                if is_horizontal == True:
                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='row '+str(self.col), y='value', is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value)
                                elif is_horizontal == False:
                                    vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='value', y='column '+str(self.row), is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value)
                            elif is_horizontal == True:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='box plot', x='value', y='column '+str(self.row), color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value)
                            elif is_horizontal == False:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='box plot', x='row '+str(self.col), y='value', color='row '+str(self.col), is_horizontal=is_horizontal, insight_type='Outliers', insight_value=insight_value)
                            else:
                                raise NotImplementedError
                        elif cur_insight_type == 'Skewness':
                            vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='histogram area', x='value', y='density', insight_type='Skewness', insight_value=insight_value)
                        elif cur_insight_type == 'Kurtosis':
                            vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='histogram bar', x='value', y='density', insight_type='Kurtosis', insight_value=insight_value)
                        elif cur_insight_type == 'Trend':
                            vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='horizon graph', x='row '+str(self.col), y='value', is_horizontal=True, insight_type='Trend', insight_value=insight_value)
                        elif cur_insight_type == 'Dominance':
                            vis_info = None
                        elif cur_insight_type == 'Top 2':
                            vis_info = None
                        elif cur_insight_type == 'Evenness':
                            vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='density plot', x='value', y='density', insight_type='Evenness', insight_value=insight_value)
                        elif cur_insight_type == 'M-Dominance':
                            if is_horizontal == True:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='row '+str(self.col), theta='value', radius='value', is_horizontal=is_horizontal, insight_type='M-Dominance', insight_value=insight_value)
                            elif is_horizontal == False:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='column '+str(self.row), theta='value', radius='value', is_horizontal=is_horizontal, insight_type='M-Dominance', insight_value=insight_value)
                            else:
                                raise NotImplementedError
                        elif cur_insight_type == 'M-Top 2':
                            if is_horizontal == True:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='row '+str(self.col), theta='value', radius='value', is_horizontal=is_horizontal, insight_type='M-Top 2', insight_value=insight_value)
                            elif is_horizontal == False:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='radial plot', color='column '+str(self.row), theta='value', radius='value', is_horizontal=is_horizontal, insight_type='M-Top 2', insight_value=insight_value)
                            else:
                                raise NotImplementedError
                        elif cur_insight_type == 'M-Evenness':
                            if is_horizontal == True:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='row '+str(self.col), y='value', color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='M-Evenness', insight_value=insight_value)
                            elif is_horizontal == False:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='bar chart', x='value', y='column '+str(self.row), color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='M-Evenness', insight_value=insight_value)
                            else:
                                raise NotImplementedError
                        elif cur_insight_type == 'Pearsonr':
                            if is_horizontal:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='multi line chart', x='row '+str(self.col), y='value', color='column '+str(self.row), is_horizontal=is_horizontal, insight_type='Pearsonr', insight_value=insight_value)
                            else:
                                vis_info = VisEncoding(pos_row=pos_row, pos_col=pos_col, vis_type='multi line chart', x='value', y='column '+str(self.row), color='row '+str(self.col), is_horizontal=is_horizontal, insight_type='Pearsonr', insight_value=insight_value)
                        if vis_info != None:
                            self.enc_list.append(vis_info)




        # if self.mask.isnull().values.any():
        #     print('reward', reward, np.abs(self.mask.values.round()) > 1e-5, (np.abs(self.mask.values.round()) > 1e-5).sum(), (np.abs(self.mask.values.round()) < 1e-5), (np.abs(self.mask.values.round()) < 1e-5).sum())
        #     raise NotImplementedError
        insight_ratio = self.bool_insight.sum()*1./self.bool_insight.shape[0]
        area_ratio = (np.abs(self.mask.values.round()) > 1e-5).sum()*1./(self.mask.shape[0]*self.mask.shape[1])
        if not self.Jain:
            Evenness_index = entropy(self.num_insight)/np.log(self.bool_insight.shape[0]) if self.num_insight.sum() > 0 else 0
            diversity = (Evenness_index**3+0.1) * np.sqrt(area_ratio) * 20
        else:
            Evenness_index = self.num_insight.sum()**2 / (self.num_insight**2).sum() / self.num_insight.shape[0] if self.num_insight.sum() > 0 else 0
            diversity = Evenness_index * np.sqrt(area_ratio) * 20


        if reward > 0:
            if self.Rcd:
                reward /= self.mask.shape[0]*self.mask.shape[1]*30
                # print('reward', self.cur_insight_type, reward, Evenness_index)
                # print('reward', self.cur_insight_type, reward, Evenness_index - self.old_ER, reward + (Evenness_index - self.old_ER))
                if not self.hand_ER:
                    if not self.end_ER:
                        if not self.Jain:
                            # reward *= (1+(Evenness_index - self.old_ER) * 10)
                            reward += (Evenness_index - self.old_ER) * self.addER
                            # reward *= (Evenness_index - self.old_ER) * 1
                            # reward = (Evenness_index - self.old_ER) * 1
                            self.old_ER = Evenness_index 
                        else:
                            # reward *= (1+(Evenness_index - self.old_ER) * 10)
                            reward += (Evenness_index - self.old_ER) * self.addER
                            # reward *= (Evenness_index - self.old_ER) * 1
                            # reward = (Evenness_index - self.old_ER) * 1
                            self.old_ER = Evenness_index
            else:
                reward /= self.mask.shape[0]*self.mask.shape[1]/self.single_zone_max_area
            # reward = 0
            # print('reward', reward, np.sqrt(insight_ratio), Evenness_index, np.sqrt(area_ratio), diversity)



        self.steps += 1
        obs = self.get_obs()
        self.episodeR += reward
        # info = EnvInfo(game_score=self.episodeR, traj_done=False, insight_ratio=0, area_ratio=0, Evenness_index=0)
        info = EnvInfo(game_score=self.episodeR, traj_done=False, insight_ratio=-1, area_ratio=-1, Evenness_index=-1)
        if done or self.steps > self.episode_step:
            if not self.hand_ER:
                if not self.Rcd:
                    reward += diversity
                if self.end_ER:
                    # if not self.Jain:
                    #     # print('reward', self.episodeR, area_ratio, Evenness_index, area_ratio * Evenness_index * 30)
                    #     reward += area_ratio * Evenness_index * 30
                    # else:
                    #     reward += area_ratio * Evenness_index * 30
                    if not self.Jain:
                        # print('reward', self.episodeR, area_ratio, Evenness_index, (area_ratio + Evenness_index) * 10)
                        reward += (area_ratio + Evenness_index * self.end_add) * 10
                    else:
                        reward += (area_ratio + Evenness_index * self.end_add) * 10
            # self.episodeR += 
            # print('episodeR', self.episodeR, np.sqrt(insight_ratio), Evenness_index, Evenness_index**3, np.sqrt(area_ratio), diversity)
            done = True
            info = EnvInfo(game_score=self.episodeR, traj_done=True, insight_ratio=insight_ratio, area_ratio=area_ratio, Evenness_index=Evenness_index)
            # print('sb_trans_cnt', self.sb_trans_cnt)
            if self.log_enc:
                path = 'test_results/0327_12_07_39/ppo/lr0001/Income_200423_8prov_end10add1ER_Step200_Rcdiv30' 
                os.makedirs(path, exist_ok=True)
                id = str(np.random.randint(1000000))
                self.mask.to_csv(path + "/eval_mask" + '_IR' + str(insight_ratio) + '_AR' + "%.5f" % area_ratio + '_EI' + "%.5f" % Evenness_index + '_' + id + ".csv", encoding='utf_8_sig')
                with open (path + "/data" + '_IR' + str(insight_ratio) + '_AR' + "%.5f" % area_ratio + '_EI' + "%.5f" % Evenness_index + '_'  + id + ".txt", 'wb') as f:
                    pickle.dump((self.table, self.enc_list), f)
                np.savetxt(path + "/num_insight" + '_IR' + str(insight_ratio) + '_AR' + "%.5f" % area_ratio + '_EI' + "%.5f" % Evenness_index + '_'  + id + ".txt", self.num_insight)
                print('reward', self.episodeR, insight_ratio, Evenness_index, area_ratio, diversity)
                # raise NotImplementedError
            # self.table.to_csv("eval_table.csv")
            # self.mask.to_csv("eval_mask.csv")
            # self.mask.to_csv("eval_mask.csv", encoding='utf_8_sig')
            # print('episodeR', self.episodeR)
            # raise NotImplementedError
        return obs, reward, done, info
    
    def close(self):
        pass
    
    def pandas2json(self, df):
        table_json = {}
        table_json['Title'] = 'test'
        table_json['MergedRegions'] = []
        table_json['TopHeaderRowsNumber'] = len(df.columns.names)
        table_json['Height'] = df.shape[0]
        table_json['LeftHeaderColumnsNumber'] = len(df.index.names)
        table_json['Width'] = df.shape[1]

        def get_json_tree(df_index, is_top=False):
            dir = {'CI': -1, "Cd": {}, "RI" :-1}
            for i, index in enumerate(df_index):
                cur_di = dir
                if isinstance(index, int):
                    if not is_top:
                        new_di = {
                                'CI': -1,
                                'Cd': {},
                                'RI': i
                            }
                    else:
                        new_di = {
                                'CI': i,
                                'Cd': {},
                                'RI': -1
                            }
                    cur_di["Cd"][index] = new_di
                else:
                    for j, ind in enumerate(index):
                        if ind in cur_di["Cd"]:
                            cur_di = cur_di["Cd"][ind]
                        else:
                            if not is_top:
                                new_di = {
                                    'CI': -1,
                                    'Cd': {},
                                    'RI': i
                                }
                            else:
                                new_di = {
                                    'CI': i,
                                    'Cd': {},
                                    'RI': -1
                                }
                            cur_di["Cd"][ind] = new_di
                            cur_di = new_di
            def clean(cur_di):
                if 'Cd' in cur_di:
                    cur_di['Cd'] = list(cur_di['Cd'].values())
                    for j in cur_di['Cd']:
                        clean(j)
            clean(dir)
            return dir
    
        table_json['LeftTreeRoot'] = get_json_tree(df.index)
        table_json['TopTreeRoot'] = get_json_tree(df.columns, is_top=True)
        table_json['Texts'] = np.array(df.values, dtype=str).tolist()
        table_json['MergedRegions'] = []
        df_index = df.index
        for j in range(table_json['LeftHeaderColumnsNumber']):
            l = r = 0
            for i in range(df.shape[0]):
                if i == 0 or df_index[i][j] == df_index[i-1][j]:
                    r = i
                else:
                    if l != r:
                        table_json['MergedRegions'].append({
                            "FirstColumn": j,
                            "LastColumn": j,
                            "FirstRow": l,
                            "LastRow": r
                        })
                    l = r = i
            if l != r:
                table_json['MergedRegions'].append({
                    "FirstColumn": j,
                    "LastColumn": j,
                    "FirstRow": l,
                    "LastRow": r
                })
        df_index = df.columns
        for j in range(table_json['TopHeaderRowsNumber']):
            l = r = 0
            for i in range(df.shape[1]):
                if i == 0 or df_index[i][j] == df_index[i-1][j]:
                    r = i
                else:
                    if l != r:
                        table_json['MergedRegions'].append({
                            "FirstColumn": l,
                            "LastColumn": r,
                            "FirstRow": j,
                            "LastRow": j
                        })
                    l = r = i
            if l != r:
                table_json['MergedRegions'].append({
                    "FirstColumn": l,
                    "LastColumn": r,
                    "FirstRow": j,
                    "LastRow": j
                })
        # import json
        # with open('ttttt.json', 'w') as f:
        #     json.dump(table_json, f)
        # raise NotImplementedError
        return table_json