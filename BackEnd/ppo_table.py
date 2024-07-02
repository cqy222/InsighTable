# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_ataripy
import argparse
from collections import deque
import os
import random
import time
import datetime
from distutils.util import strtobool
import copy

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from htable.eva import EvaENV


# from stable_baselines3.common.atari_wrappers import (  # isort:skip
#     ClipRewardEnv,
#     EpisodicLifeEnv,
#     FireResetEnv,
#     MaxAndSkipEnv,
#     NoopResetEnv,
# )


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=17,
        help="seed of the experiment")
    # 17        5
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="Console",
        help="the id of the environment")
    parser.add_argument("--data-name", type=str, default="Console5")
    parser.add_argument("--total-timesteps", type=int, default=15000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2e-4,#2e-4
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,#32
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,#64
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,#16
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,#0.2
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,#0.01
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name, data_name='Console5', eval=False, cur_df=None, cur_mask=None):
    def thunk():
        env = EvaENV(data_name, eval, cur_df, cur_mask, seed)
        # env = EvaENV(eval)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.mlp = layer_init(nn.Linear(in_features, out_features))

    def forward(self, x, adj):
        x = self.mlp(x)
        x = torch.matmul(adj, x)
        return x

class GCN(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCN, self).__init__()
        self.layer = 1
        self.conv = nn.ModuleList([GraphConvolution(in_features, out_features)] + [GraphConvolution(out_features, out_features) for _ in range(self.layer-1)])
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x, adj):
        adj += torch.eye(adj.size(-1), device=adj.device)
        D_minus_sqrt = torch.diag_embed(torch.sum(adj, dim=-1)).inverse().sqrt()
        A_hat = D_minus_sqrt.matmul(adj).matmul(D_minus_sqrt)
        # A_hat = adj
        # print('A_hat', A_hat.shape, A_hat)
        # print('GCN', x.shape, A_hat.shape)
        for conv in self.conv:
            x = conv(x, A_hat)
            x = self.relu(x)
            # x = self.tanh(x)
        return x

class Table(nn.Module):
    def __init__(self, envs, fc_sizes, rnd=False):
        super().__init__()
        self.fc_sizes = fc_sizes
        self.gnn_conv = GCN(envs.single_observation_space["x"].shape[1], self.fc_sizes)
        self.clean_mlp = nn.Sequential(layer_init(nn.Linear(envs.single_observation_space["datas"].shape[0]+self.fc_sizes, self.fc_sizes)), nn.ReLU(inplace = True))
    
    def forward(self, input):
        x = input[0]
        edge_index = input[1]
        datas = input[2]
        pre_shape = x.shape[:len(x.shape)-2]

        x = x.view(-1, *x.shape[-2:])
        edge_index = edge_index.view(-1, *edge_index.shape[-2:])
        datas = datas.view(-1, *datas.shape[-1:])


        B = datas.shape[0]
        p0_1 = datas[:,0:1]
        p1_2 = datas[:,1:2]
        p1 = int((datas[:,1].max()*100).round().item())
        p3 = datas[:,3]
        p4 = datas[:,4]
        p5 = datas[:,5]
        p6 = datas[0,6].round().int()
        p7 = datas[0,7].round().int()


        pre_datas = datas[:,:8+10]
        datas = datas[:,8+10:]


        x = self.gnn_conv(x, edge_index).view(B, -1, self.fc_sizes)
        # print('p5', x.shape, p5.shape, p5)
        mask_p5 = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(B, -1) < p5.unsqueeze(1)
        # print('mask_p5', x.shape, mask_p5.shape, mask_p5[0], p5)
        x = (x * mask_p5.unsqueeze(-1)).sum(1) / mask_p5.sum(1, True)
        # print('====x', x.shape, datas.shape)

        tmp = torch.cat((pre_datas, datas, x), dim=-1)
        # print('hptest', tmp.shape)
        fc_out = self.clean_mlp(torch.cat((pre_datas, datas, x), dim=-1))

        return fc_out, (p1, p0_1, p1_2)

class Agent(nn.Module):
    def __init__(self, envs, fc_sizes=256):
        super().__init__()
        self.fc_sizes = fc_sizes
        self.network = Table(envs, self.fc_sizes)
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.fc_sizes, self.fc_sizes)),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            layer_init(nn.Linear(self.fc_sizes, envs.single_action_space.n), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.fc_sizes, self.fc_sizes)),
            nn.Tanh(),
            # nn.ReLU(inplace=True),
            layer_init(nn.Linear(self.fc_sizes, 1), std=1),
        )

    def get_value(self, x):
        hidden, _ = self.network(x)
        v = self.critic(hidden)
        # print('hidden', hidden.shape, hidden[0][:10], v[0])
        return v

    def get_action_and_value(self, x, action=None):
        hidden, mask = self.network(x)
        # print('hidden', hidden)
        logits = self.actor(hidden)
        

        # p1, p0_1, p1_2 = mask
        # mask = torch.ones_like(logits, requires_grad = False)
        # if p1 > 0:
        #     mask[:,:p1] = 0
        #     mask = (mask - p0_1).abs()# * ((p1_2*100).round() > 1)
        # else:
        #     mask *= 0
        #     raise NotImplementedError
        # if p1 == 0:
        #     print('mask', mask, mask.shape)
        #     raise NotImplementedError
        # mask = mask.round()
        # # print('******mask', mask.shape, mask)
        # # print('pre_logits', logits.shape, logits[0])
        # logits += mask * (-1e8) #mask here

        # print('logits', logits)
        probs = Categorical(logits=logits)

        # probs = Categorical(probs=torch.nn.functional.softmax(logits, -1))
        # ent = probs.entropy()
        # print('entropy', ent.shape, ent[0])
        # raise NotImplementedError
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


# def pre_update_next_encoding_dataset():
# tmp_list = []

def update_next_encoding_dataset(insight_list, pdselectedDF):
    # global tmp_list
    # if len(tmp_list) > 0:
    #     cur_id = np.random.randint(len(tmp_list))
    #     print("cur_idcur_idcur_id", cur_id)
    #     return tmp_list[cur_id]


    print('update_next_encoding_dataset')
    # print('pdselectedDF', pdselectedDF)
    # print('insight_list', insight_list)

    mask = copy.deepcopy(pdselectedDF) * 0.
    print("====================================")
    for insight in insight_list:
        print("------every", insight)
        #不是selectedArea，而是hp_pos, pos_row, pos_col
        selectedArea = insight["selectedArea"]
        recommendArea = insight["recommendArea"]
        insight_type = insight["insight_type"]
        # mask.iloc[selectedArea["top"] : selectedArea["bottom"] + 1, selectedArea["left"] : selectedArea["right"] + 1] = 1
        # eval_result('Console')

        type_mp = {
            "Unit Visualization":1,
            "unit visualization":1,
            "unit":1,
            "line chart":2,
            "strip plot":3,
            "box plot":4,
            "bar chart":5,
            "stacked bar chart":5,
            "horizon graph":6,
            "scatterplot":7,
            "parallel coordinate plot":8,
            "pie chart":9,
            "multi series line chart":10,
            "multi line chart":10,
            "density plot":11,
            "radial plot":12,
            "histogram bar":13,
            "histogram area":14,
            "ranged dot plot":15,
            "radial plot":15,
            "2d histogram heatmap":16,
        }
        # self._insight_type_single = ['Outliers', 'Skewness', 'Kurtosis', 'Trend']
        # self._insight_type_multiple = ['Pearsonr', 'M-Dominance', 'M-Top 2', 'M-Evenness']
        insight_type = insight_type.lower()
        if insight_type in type_mp:
            # print("type_mp[insight_type]", type_mp[insight_type])
            mask.iloc[selectedArea["top"] : selectedArea["bottom"] + 1, selectedArea["left"] : selectedArea["right"] + 1] = type_mp[insight_type]
        else:
            print("insight_type", insight_type)
            raise NotImplementedError
        for area in recommendArea:
            mask.iloc[area["top"] : area["bottom"] + 1, area["left"] : area["right"] + 1] = - type_mp[insight_type]
    


    # print("maskmaskmask", mask.shape, mask.columns, mask.index)
    data_name = "Console5"
    files = os.listdir('eval_models')
    run_name = 'null'
    # print('eval_result_files', files)
    ok_list = []
    for file in files:
        if data_name in file:
            # run_name = file
            ok_list.append(file)
            # break
    # if run_name == 'null':
    if len(ok_list) == 0:
        raise NotImplementedError('No modules found!')
    run_name = ok_list[np.random.randint(len(ok_list))]
    model_path = 'eval_models/' + run_name + '/agent.pth'
    print('hpmodel_path', model_path, os.path.exists(model_path))
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent = torch.load(model_path, map_location=device)
    

    
    envs = gym.vector.SyncVectorEnv(
    # envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, run_name, eval=True, cur_df=pdselectedDF, cur_mask=mask) for i in range(32)],
    )
    # envs = EvaENV(data_name, eval=True)
    ob = envs.reset()
    
    # envs.table.to_csv('envs_table.csv', encoding='utf_8_sig')
    # envs.mask.to_csv('envs_mask.csv', encoding='utf_8_sig')
    pdselectedDF.to_csv('pdselectedDF.csv', encoding='utf_8_sig')
    mask.to_csv('mask.csv', encoding='utf_8_sig')
    # print("envs.tableenvs.table", envs.envs[0].table)
    # envs.table = pdselectedDF
    # envs.mask = mask
    # for i in range(len(envs.envs)):
    #     envs.envs[i].table = pdselectedDF
    #     envs.envs[i].mask = mask
    # ob = envs.get_obs()
    # ob = envs.step(np.zeros(32))[0]
    # print("obobobobobob", type(ob), len(ob), ob,)
    # raise NotImplementedError
    # mid change here
    next_obs_x = torch.Tensor(ob["x"]).to(device)
    next_obs_edge_index = torch.Tensor(ob["edge_index"]).to(device)
    next_obs_datas = torch.Tensor(ob["datas"]).to(device)
    done_cnt = 0
    alter_enclist = []
    alter_reward = []

    for step in range(0, args.total_timesteps):
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value((next_obs_x, next_obs_edge_index, next_obs_datas))
        next_obs_dict, reward, done, info = envs.step(action.cpu().numpy())
        indices = np.where(done)[0]


        # honor's version
        if len(indices) > 0:
            alter_enclist.extend(info["enc_list"][indices])
            alter_reward.extend(info["reward"][indices])
            if len(alter_enclist) >= 32:
                break
            raise NotImplementedError

        # # sys's version
        # if len(indices) > 0:
        #     info = np.array(info)
        #     for cur_info in info[indices]:
        #         print("cur_infocur_info", type(cur_info), cur_info)
        #         alter_enclist.append(cur_info["enc_list"])
        #         alter_reward.append(cur_info["reward"])
        #     if len(alter_enclist) >= 32:
        #         break
        #     raise NotImplementedError


        next_obs_x = torch.Tensor(next_obs_dict["x"]).to(device)
        next_obs_edge_index = torch.Tensor(next_obs_dict["edge_index"]).to(device)
        next_obs_datas = torch.Tensor(next_obs_dict["datas"]).to(device)


    print("alter_rewardalter_reward", alter_reward)
    # for hpp in alter_enclist:
    #     print("hpphpphpphpphpp")
    #     for _ in hpp:
    #         print("hhhhhhhhhh", _.pos_row, _.pos_col, _.vis_type, _.rec_list)
    # tmp_list = alter_enclist
    indices = np.random.choice(np.where(alter_reward >= np.max(alter_reward)-0.05)[0])
    # indices = np.argmax(alter_reward)
    alter_reward = sorted(alter_reward, reverse=True)
    # print("indicesindices", np.where(alter_reward >= np.max(alter_reward)-0.1)[0], indices)
    result = alter_enclist[indices]
    # print("tttttttttttttttt", alter_reward, alter_reward[indices], type(result), result)
    # raise NotImplementedError
    return result

def eval_result(data_name):
    print("os.getcwd", os.getcwd())
    files = os.listdir('eval_models')
    run_name = 'null'
    print('eval_result_files', files)
    for file in files:
        if data_name in file:
            run_name = file
            # break
    if run_name == 'null':
        raise NotImplementedError('No modules found!')
    
    model_path = 'eval_models/' + run_name + '/agent.pth'
    print('hpmodel_path', model_path, os.path.exists(model_path))

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
#  .SyncVectorEnv
    # envs = gym.vector.SyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, False, run_name, eval=True) for i in range(32)],
    # )
    envs = EvaENV(data_name, eval=True)
    agent = torch.load(model_path, map_location=device)

    ob = envs.reset()
    next_obs_x = torch.Tensor(ob["x"]).to(device)
    next_obs_edge_index = torch.Tensor(ob["edge_index"]).to(device)
    next_obs_datas = torch.Tensor(ob["datas"]).to(device)
    done_cnt = 0

    for step in range(0, args.total_timesteps):
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value((next_obs_x, next_obs_edge_index, next_obs_datas))
        next_obs_dict, reward, done, info = envs.step(action.cpu().numpy())
        # done_cnt += done.sum()
        if done:
            done_cnt += 1
            ob = envs.reset()
            next_obs_x = torch.Tensor(ob["x"]).to(device)
            next_obs_edge_index = torch.Tensor(ob["edge_index"]).to(device)
            next_obs_datas = torch.Tensor(ob["datas"]).to(device)
        if done_cnt > 10:
            break
        next_obs_x = torch.Tensor(next_obs_dict["x"]).to(device)
        next_obs_edge_index = torch.Tensor(next_obs_dict["edge_index"]).to(device)
        next_obs_datas = torch.Tensor(next_obs_dict["datas"]).to(device)

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{datetime.datetime.now().strftime('%m%d_%H_%M_%S')}__{args.exp_name}"
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    # envs = gym.vector.AsyncVectorEnv(
    #     [make_env(args.env_id, args.seed + i, i, False, run_name) for i in range(args.num_envs)],
    #     context = 'spawn'
    # )
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, run_name, args.data_name, eval=True) for i in range(args.num_envs)],
        # [make_env(args.env_id, args.seed + i, i, False, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, fc_sizes=256).to(device)
    # agent = torch.load('runs/BreakoutNoFrameskip-v4__ppo_big_income_lenall_bert64_step50_wrnn_nooutly_ly1_drop0_worec_went_clip1_rdiv1_sortx_moveaction0_lr2e4_mini4_num8_Sync_steps64_lyNorm__1__1692073271/agent.pkl')
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    shape_x = (args.num_steps, args.num_envs) + envs.single_observation_space["x"].shape
    shape_edge_index = (args.num_steps, args.num_envs) + envs.single_observation_space["edge_index"].shape
    shape_datas = (args.num_steps, args.num_envs) + envs.single_observation_space["datas"].shape
    obs_x = torch.zeros(shape_x).to(device)
    obs_edge_index = torch.zeros(shape_edge_index).to(device)
    obs_datas = torch.zeros(shape_datas).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=args.num_envs*2)
    
    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # next_obs = torch.Tensor(envs.reset()).to(device)
    ob = envs.reset()
    next_obs_x = torch.Tensor(ob["x"]).to(device)
    next_obs_edge_index = torch.Tensor(ob["edge_index"]).to(device)
    next_obs_datas = torch.Tensor(ob["datas"]).to(device)

    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            # obs[step] = next_obs
            obs_x[step] = next_obs_x
            obs_edge_index[step] = next_obs_edge_index
            obs_datas[step] = next_obs_datas
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # action, logprob, _, value = agent.get_action_and_value(next_obs)
                action, logprob, _, value = agent.get_action_and_value((obs_x[step], obs_edge_index[step], obs_datas[step]))
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            # next_obs, reward, done, info = envs.step(action.cpu().numpy())
            next_obs_dict, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            # next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            next_obs_x = torch.Tensor(next_obs_dict["x"]).to(device)
            next_obs_edge_index = torch.Tensor(next_obs_dict["edge_index"]).to(device)
            next_obs_datas = torch.Tensor(next_obs_dict["datas"]).to(device)
            next_done = torch.Tensor(done).to(device)

            # for item in info:
            #     if "episode" in item.keys():
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         break

            # for idx, item in enumerate(info):
            #     if "episode" in item.keys():
            #         avg_returns.append(item["episode"]["r"])
            #         epi_ret = np.average(avg_returns)
            #         print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
            #         writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
            #         # print('item', item.keys())
            #         # item dict_keys(['reward', 'insight_ratio', 'area_ratio', 'Evenness_index', 'terminated', 'episode', 'terminal_observation'])
            #         writer.add_scalar("charts/insight_ratio", item["insight_ratio"], global_step)
            #         writer.add_scalar("charts/area_ratio", item["area_ratio"], global_step)
            #         writer.add_scalar("charts/Evenness_index", item["Evenness_index"], global_step)
            #         writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
            #         break

            # if "episode" in info.keys():
            #     print("infoinfo", type(info))
            #     print("infoepisode", type(info["episode"]))
            #     print("infoepisoder", type(info["episode"]["r"]))
            #     avg_returns.append(info["episode"]["r"])
            #     epi_ret = np.average(avg_returns)
            #     print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
            #     writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            #     writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
            #     # print('info', info.keys())
            #     # info dict_keys(['reward', 'insight_ratio', 'area_ratio', 'Evenness_index', 'terminated', 'episode', 'terminal_observation'])
            #     writer.add_scalar("charts/insight_ratio", info["insight_ratio"], global_step)
            #     writer.add_scalar("charts/area_ratio", info["area_ratio"], global_step)
            #     writer.add_scalar("charts/Evenness_index", info["Evenness_index"], global_step)
            #     writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)

        # bootstrap value if not done
        with torch.no_grad():
            # next_value = agent.get_value(next_obs).reshape(1, -1)
            next_value = agent.get_value((next_obs_x, next_obs_edge_index, next_obs_datas)).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_obs_x = obs_x.reshape((-1,) + envs.single_observation_space["x"].shape)
        b_obs_edge_index = obs_edge_index.reshape((-1,) + envs.single_observation_space["edge_index"].shape)
        b_obs_datas = obs_datas.reshape((-1,) + envs.single_observation_space["datas"].shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    (b_obs_x[mb_inds], b_obs_edge_index[mb_inds], b_obs_datas[mb_inds]), b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                # print('newlogprob', newlogprob, b_logprobs[mb_inds])
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        # print('var_y', var_y)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/mb_advantages_mean", mb_advantages.mean().item(), global_step)
        writer.add_scalar("losses/mb_advantages_max", mb_advantages.max().item(), global_step)
        writer.add_scalar("losses/mb_advantages_min", mb_advantages.min().item(), global_step)
        writer.add_scalar("losses/mb_advantages_var", mb_advantages.var().item(), global_step)
        writer.add_scalar("losses/ratio_mean", ratio.mean().item(), global_step)
        writer.add_scalar("losses/ratio_max", ratio.max().item(), global_step)
        writer.add_scalar("losses/ratio_min", ratio.min().item(), global_step)
        writer.add_scalar("losses/ratio_var", ratio.var().item(), global_step)

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if update % 10 == 0:
        # if update % 1 == 0:
            os.makedirs(f"eval_models/{run_name}", exist_ok=True)
            print('torchsave: ', f"eval_models/{run_name}/agent.pth")
            torch.save(agent, f"eval_models/{run_name}/agent.pth")
    envs.close()
    writer.close()
