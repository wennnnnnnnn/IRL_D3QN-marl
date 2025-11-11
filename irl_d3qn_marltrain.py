import scipy.io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Environment_marl
import random
import os
from replay_memory import ReplayMemory

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ################## SETTINGS ######################
up_lanes = [i / 2.0 for i in
            [3.5 / 2, 3.5 / 2 + 3.5, 250 + 3.5 / 2, 250 + 3.5 + 3.5 / 2, 500 + 3.5 / 2, 500 + 3.5 + 3.5 / 2]]
down_lanes = [i / 2.0 for i in
              [250 - 3.5 - 3.5 / 2, 250 - 3.5 / 2, 500 - 3.5 - 3.5 / 2, 500 - 3.5 / 2, 750 - 3.5 - 3.5 / 2,
               750 - 3.5 / 2]]
left_lanes = [i / 2.0 for i in
              [3.5 / 2, 3.5 / 2 + 3.5, 433 + 3.5 / 2, 433 + 3.5 + 3.5 / 2, 866 + 3.5 / 2, 866 + 3.5 + 3.5 / 2]]
right_lanes = [i / 2.0 for i in
               [433 - 3.5 - 3.5 / 2, 433 - 3.5 / 2, 866 - 3.5 - 3.5 / 2, 866 - 3.5 / 2, 1299 - 3.5 - 3.5 / 2,
                1299 - 3.5 / 2]]
width = 750 / 2
height = 1298 / 2
label = 'marl_irl_dueling_dqn_model'  # Updated model name
n_veh = 4  # For the number of the vehicles in each direction remain same, n_veh % 4 == 0
n_neighbor = 1
n_RB = n_veh
# Environment Parameters
env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()  # initialize parameters in env
n_episode = 3000
n_step_per_episode = int(env.time_slow / env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.85 * n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode * 4
n_episode_test = 100  # test episodes
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120

# 添加学习率调度和奖励权重系数
REWARD_WEIGHT = 0.7  # 调整奖励网络的影响权重
LR_DECAY_STEP = 500  # 学习率衰减步数


######################################################

def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10) / 35
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]],
                :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10) / 35
    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60
    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80) / 60.0
    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])
    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs,
                           time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


n_input_size = len(get_state(env=env))
n_output_size = n_RB * len(env.V2V_power_dB_List)


class DuelingDQN(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size):
        super(DuelingDQN, self).__init__()
        # Shared feature layers
        self.feature_layer = nn.Sequential(
            nn.Linear(input_size, n_hidden1),
            nn.ReLU(),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU()
        )

        # State value stream
        self.value_stream = nn.Sequential(
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3, 1)  # Output a single state value
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(n_hidden2, n_hidden3),
            nn.ReLU(),
            nn.Linear(n_hidden3, output_size)  # Output advantage for each action
        )

        # Initialize weights with small random values
        for layer in self.feature_layer:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.1)

        for layer in self.value_stream:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.1)

        for layer in self.advantage_stream:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(0, 0.1)

    def forward(self, x):
        features = self.feature_layer(x)

        # Calculate state value
        value = self.value_stream(features)

        # Calculate advantages
        advantages = self.advantage_stream(features)

        # Combine value and advantages to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values


class RewardNet(nn.Module):
    def __init__(self, state_size):
        super(RewardNet, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        # 初始化权重更保守
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc3.weight.data.normal_(0, 0.01)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Agent:
    def __init__(self, memory_entry_size):
        self.discount = 0.99  # 改为0.99以提高稳定性
        self.double_q = True  # 使用Double Q-learning提高稳定性
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)
        self.model = DuelingDQN(n_input_size, n_hidden_1, n_hidden_2, n_hidden_3, n_output_size).to(device)
        self.target_model = DuelingDQN(n_input_size, n_hidden_1, n_hidden_2, n_hidden_3, n_output_size).to(device)
        self.reward_net = RewardNet(n_input_size).to(device)  # Added Reward Network for IRL

        self.target_model.eval()

        # 使用Adam优化器提高稳定性
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.reward_optimizer = torch.optim.Adam(self.reward_net.parameters(), lr=0.0005)

        # 学习率调度
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=LR_DECAY_STEP, gamma=0.9)
        self.reward_scheduler = torch.optim.lr_scheduler.StepLR(self.reward_optimizer, step_size=LR_DECAY_STEP,
                                                                gamma=0.95)

        # 使用Huber损失函数提高稳定性
        self.loss_func = nn.SmoothL1Loss()

        # 记录一些训练统计信息
        self.train_iter = 0
        self.reward_scale = 1.0  # 初始奖励缩放

    def predict(self, s_t, ep=0.):
        n_power_levels = len(env.V2V_power_dB_List)
        if random.random() > ep:
            with torch.no_grad():
                q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device))
                return q_values.max(1)[1].item()
        else:
            return random.choice(range(n_output_size))

    def Q_Learning_mini_batch(self):
        self.train_iter += 1

        # 采样批量数据
        batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = self.memory.sample()

        # 转换为张量
        action = torch.LongTensor(batch_action).to(device)
        state = torch.FloatTensor(np.float32(batch_s_t)).to(device)
        next_state = torch.FloatTensor(np.float32(batch_s_t_plus_1)).to(device)
        env_reward = torch.FloatTensor(batch_reward).to(device)

        # 每500次迭代调整一次奖励缩放
        if self.train_iter % 500 == 0:
            with torch.no_grad():
                predicted_rewards = self.reward_net(state).squeeze()
                reward_mean = predicted_rewards.mean().item()
                reward_std = predicted_rewards.std().item()
                if reward_std > 0:
                    # 动态调整缩放以保持奖励有合理范围
                    self.reward_scale = min(1.0, max(0.1, 1.0 / (reward_std * 5)))

        # 首先训练奖励网络
        self.reward_optimizer.zero_grad()
        reward_pred = self.reward_net(state).squeeze()

        # 使用环境奖励作为指导，同时允许奖励网络学习更丰富的表示
        # 使用均方误差引导奖励网络与环境奖励保持一定相关性
        reward_guidance_loss = F.mse_loss(reward_pred, env_reward.squeeze())

        # 同时鼓励奖励网络产生区分性强的奖励值（最大化奖励的方差）
        reward_diversity_loss = -torch.var(reward_pred) * 0.1

        # 综合奖励网络损失
        reward_loss = reward_guidance_loss + reward_diversity_loss

        reward_loss.backward()
        # 梯度裁剪防止奖励网络梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.reward_net.parameters(), 1.0)
        self.reward_optimizer.step()

        # 然后训练Q网络
        # 使用奖励网络的预测进行Q学习
        with torch.no_grad():
            learned_rewards = self.reward_net(state).squeeze() * self.reward_scale

            # 将环境奖励与学习的奖励结合
            combined_rewards = env_reward.squeeze() + REWARD_WEIGHT * learned_rewards

            if self.double_q:
                # Double Q-learning
                next_action = self.model(next_state).max(1)[1]
                next_q_values = self.target_model(next_state)
                next_q_value = next_q_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            else:
                next_q_value = self.target_model(next_state).max(1)[0]

            expected_q_value = combined_rewards + self.discount * next_q_value

        # 计算当前Q值
        q_values = self.model(state)
        q_acted = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # 计算Q网络损失
        q_loss = self.loss_func(q_acted, expected_q_value)

        # 反向传播并更新Q网络
        self.optimizer.zero_grad()
        q_loss.backward()
        # 梯度裁剪防止Q网络梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # 更新学习率
        self.scheduler.step()
        self.reward_scheduler.step()

        # 记录总损失
        total_loss = q_loss.item() + reward_loss.item()
        return total_loss

    def update_target_network(self):
        # 使用软更新来提高稳定性
        tau = 0.01  # 软更新系数
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.model.state_dict(), model_path + '.ckpt')
        torch.save(self.target_model.state_dict(), model_path + '_t.ckpt')
        torch.save(self.reward_net.state_dict(), model_path + '_reward.ckpt')

    def load_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        self.model.load_state_dict(torch.load(model_path + '.ckpt'))
        self.target_model.load_state_dict(torch.load(model_path + '_t.ckpt'))
        self.reward_net.load_state_dict(torch.load(model_path + '_reward.ckpt'))


# ----------------------------------------------------------------------------
print(device)
agents = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)

# ----------------------------Training----------------------------------------
record_reward = np.zeros([n_episode * n_step_per_episode, 1])
record_loss = []

# 添加一个预热阶段，填充经验回放缓冲区
print("Warming up experience replay buffer...")
env.renew_positions()
env.renew_neighbor()
env.renew_channel()
env.renew_channels_fastfading()

# 预热阶段
warmup_episodes = 50
for i_episode in range(warmup_episodes):
    env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
    env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
    env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

    for i_step in range(n_step_per_episode):
        state_old_all = []
        action_all = []
        action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

        for i in range(n_veh):
            for j in range(n_neighbor):
                state = get_state(env, [i, j], i_episode / warmup_episodes, 1.0)
                state_old_all.append(state)
                action = agents[i * n_neighbor + j].predict(state, 1.0)  # 使用高探索率
                action_all.append(action)

                action_all_training[i, j, 0] = action % n_RB
                action_all_training[i, j, 1] = int(np.floor(action / n_RB))

        action_temp = action_all_training.copy()
        train_reward = env.act_for_training(action_temp)

        env.renew_channels_fastfading()
        env.Compute_Interference(action_temp)

        for i in range(n_veh):
            for j in range(n_neighbor):
                state_old = state_old_all[n_neighbor * i + j]
                action = action_all[n_neighbor * i + j]
                state_new = get_state(env, [i, j], i_episode / warmup_episodes, 1.0)

                agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, action)

    if i_episode % 10 == 0:
        print(f"Warmup episode {i_episode}/{warmup_episodes}")

print("Starting main training...")
avg_rewards = []  # 用于跟踪奖励移动平均

for i_episode in range(n_episode):
    if i_episode < epsi_anneal_length:
        epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
    else:
        epsi = epsi_final

    episode_rewards = []

    if i_episode % 100 == 0:
        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        env.renew_channel()  # update channel slow fading
        env.renew_channels_fastfading()  # update channel fast fading

    env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
    env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
    env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

    for i_step in range(n_step_per_episode):
        time_step = i_episode * n_step_per_episode + i_step
        state_old_all = []
        action_all = []
        action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')

        for i in range(n_veh):
            for j in range(n_neighbor):
                state = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                state_old_all.append(state)
                action = agents[i * n_neighbor + j].predict(state, epsi)
                action_all.append(action)

                action_all_training[i, j, 0] = action % n_RB  # chosen RB
                action_all_training[i, j, 1] = int(np.floor(action / n_RB))  # power level

        # All agents take actions simultaneously, obtain shared reward, and update the environment.
        action_temp = action_all_training.copy()
        train_reward = env.act_for_training(action_temp)
        record_reward[time_step] = train_reward
        episode_rewards.append(train_reward)

        env.renew_channels_fastfading()
        env.Compute_Interference(action_temp)

        for i in range(n_veh):
            for j in range(n_neighbor):
                state_old = state_old_all[n_neighbor * i + j]
                action = action_all[n_neighbor * i + j]
                state_new = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)

                # add entry to this agent's memory
                agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, action)

                # training this agent
                if time_step % mini_batch_step == mini_batch_step - 1:
                    loss_val_batch = agents[i * n_neighbor + j].Q_Learning_mini_batch()
                    record_loss.append(loss_val_batch)

                # 更频繁地更新目标网络
                if time_step % (target_update_step // 4) == (target_update_step // 4) - 1:
                    agents[i * n_neighbor + j].update_target_network()

    # 计算并显示每个episode的平均奖励
    avg_episode_reward = sum(episode_rewards) / len(episode_rewards)
    avg_rewards.append(avg_episode_reward)

    # 计算移动平均（最近10个episode）
    recent_avg = sum(avg_rewards[-10:]) / min(len(avg_rewards), 10)

    if i_episode % 10 == 0:
        # 打印训练进度和状态
        print(f"Episode: {i_episode}/{n_episode}, Epsilon: {epsi:.4f}, "
              f"Avg Reward: {avg_episode_reward:.4f}, Moving Avg(10): {recent_avg:.4f}")

    # 每100个episodes保存一次模型
    if i_episode % 100 == 0 and i_episode > 0:
        for i in range(n_veh):
            for j in range(n_neighbor):
                model_path = label + '/checkpoint_' + str(i_episode) + '_agent_' + str(i * n_neighbor + j)
                agents[i * n_neighbor + j].save_models(model_path)

print('Training Done. Saving models...')
for i in range(n_veh):
    for j in range(n_neighbor):
        model_path = label + '/agent_' + str(i * n_neighbor + j)
        agents[i * n_neighbor + j].save_models(model_path)

current_dir = os.path.dirname(os.path.realpath(__file__))
reward_path = os.path.join(current_dir, "model/" + label + '/reward.mat')
scipy.io.savemat(reward_path, {'reward': record_reward})

record_loss = np.asarray(record_loss).reshape((-1, n_veh * n_neighbor))
loss_path = os.path.join(current_dir, "model/" + label + '/train_loss.mat')
scipy.io.savemat(loss_path, {'train_loss': record_loss})

# 保存平均奖励数据用于后续分析
avg_rewards_path = os.path.join(current_dir, "model/" + label + '/avg_rewards.mat')
scipy.io.savemat(avg_rewards_path, {'avg_rewards': np.array(avg_rewards)})