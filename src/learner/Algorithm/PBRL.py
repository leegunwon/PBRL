import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from src.common.Parameters import *
from src.learner.common.ReplayBuffer import *
from src.learner.common.RewardModel import RewardModel
from src.simulator.Simulator import Simulator
from src.learner.common.Hyperparameters import *
from src.learner.common.Qnet import Qnet


class PBRL:
    print("PBRL on")
    # reward model 바꿀 점 label 데이터를 바탕으로 학습 데이터 만들기
    reward_model = RewardModel(
        ds=Hyperparameters.input_layer,
        da=Hyperparameters.output_layer,
        lr=Hyperparameters.reward_lr,
        max_size=Hyperparameters.episode,
        size_sample_action=Hyperparameters.size_sample_action)

    @classmethod
    def train(cls, q, q_target, memory, optimizer):
        for i in range(10):
            s, a, r, s_prime, done_mask = memory.sample(Hyperparameters.batch_size)
            q_out = q(s)
            q_a = q_out.gather(1, a)
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
            # print(max_q_prime.shape)
            target = r + Hyperparameters.gamma * max_q_prime * done_mask
            loss = F.smooth_l1_loss(q_a, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    @classmethod
    def main(cls, count):
        """
        찾아야 할 요소는 언제 train_reward를 수행하는 지
        어떻게 loss를 계산하는지
        action 선택은 어떤식으로 하는지
        :return:
        """
        # 모델 불러오는 코드 필요함
        #
        env = Simulator
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer(Hyperparameters.buffer_limit)
        makespan_list = []
        score_list = []
        util_list = []
        optimizer = optim.Adam(q.parameters(), lr=Hyperparameters.learning_rate)
        save_directory = f"{pathConfig.reinforcement_model_params_path}"
        # model load
        cls.load_reward_model(count)

        if os.path.exists(f"{save_directory}{os.sep}{count}q_net_param.pt") and (Hyperparameters.mode != 1):
            params = torch.load(f"{save_directory}{os.sep}{count}q_net_param.pt")
            q.load_state_dict(params)

        if os.path.exists(f"{save_directory}{os.sep}{-1}q_net_param.pt") and (Hyperparameters.mode == 1):
            params = torch.load(f"{save_directory}{os.sep}{-1}q_net_param.pt")
            q.load_state_dict(params)
        # cls.learn_reward()

        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0.01, 0.8 - 0.9/Hyperparameters.episode * n_epi)
            s = env.reset(Parameters.datasetId)
            done = False
            score = 0.0
            while not done:
                a = q.sample_action(torch.from_numpy(s).float(), epsilon)

                s_prime, r, done = env.step(a) # r은 그냥 0으로 설정
                inputs = np.concatenate([s, np.array([a])])
                r = cls.reward_model.r_hat(inputs)[0]
                done_mask = 0.0 if done else 1.0

                # data 생성할 때만 사용
                if (Hyperparameters.mode == 1 or Hyperparameters.mode == 4):
                    cls.reward_model.add_data(s, a, done)

                if (done == False and Hyperparameters.mode == 4):
                    memory.put([s, a, r, s_prime, done_mask])
                s = s_prime
                score += r

            if (memory.size() > 100 and Hyperparameters.mode == 4):
                cls.train(q, q_target, memory, optimizer)

            if (n_epi%10==0):
                makespan_list, util_list, score_list = cls.script_performance(env, n_epi, epsilon, memory, score,
                                                                                 True, makespan_list, util_list,
                                                                                 score_list)

        cls.reward_model.data_save()

        if Hyperparameters.mode != 1:
            params = q.state_dict()
            file_name = f"{count}q_net_param.pt"
            file_path = os.path.join(save_directory, file_name)
            torch.save(params, file_path)

            score_list = cls.normalize(score_list)

            fig = make_subplots(rows=1, cols=2)
            df = pd.DataFrame({'x': list(range(1, len(score_list)+1)), 'score': score_list})
            fig1 = px.scatter(df, x='x', y='score', trendline='ols', title='Score')
            df = pd.DataFrame({'score': score_list, 'util': util_list})
            fig2 = px.scatter(df, x='score', y='util', trendline='ols', title='Score vs Util Relationship')
            for trace in fig1.data:
                fig.add_trace(trace, row=1, col=1)
            for trace in fig2.data:
                fig.add_trace(trace, row=1, col=2)

            # Trendline에서의 예측 값 계산
            trendline_x = fig2.data[1]['x']
            trendline_y = fig2.data[1]['y']
            predictions = np.interp(df['score'], trendline_x, trendline_y)

            # 첫 번째 subplot의 x축 텍스트 크기 조정
            fig.update_xaxes(tickfont=dict(size=40), row=1, col=1)
            fig.update_xaxes(tickfont=dict(size=40), row=1, col=2)
            fig.update_yaxes(tickfont=dict(size=40), row=1, col=1)
            fig.update_yaxes(tickfont=dict(size=40), row=1, col=2)
            # 평균 값 계산
            mean_util = np.mean(df['util'])

            # Total Sum of Squares (SST) 계산
            sst = np.sum((df['util'] - mean_util) ** 2)

            # Residual Sum of Squares (SSR) 계산
            ssr = np.sum((df['util'] - predictions) ** 2)

            # R² 값 계산
            r_squared = 1 - (ssr / sst)
            fig.update_layout(title=f"r_squared : {r_squared}")
            # print("R² value: " + str(r_squared))
            # fig.show()
            fig.write_html(
                f"{pathConfig.PBRL_result_chart_path}{os.sep}{count}_score.html")

            return r_squared

    @classmethod
    def evaluate(cls, count):
        """
        찾아야 할 요소는 언제 train_reward를 수행하는 지
        어떻게 loss를 계산하는지
        action 선택은 어떤식으로 하는지
        :return:
        """
        # 모델 불러오는 코드 필요함
        #
        env = Simulator
        util_list = []
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer(Hyperparameters.buffer_limit)
        score = 0.0
        max_uitl = 0.92
        save_directory = f"{pathConfig.reinforcement_model_params_path}"
        # model load
        if os.path.exists(f"{save_directory}{os.sep}{count}q_net_param.pt"):
            params = torch.load(f"{save_directory}{os.sep}{count}q_net_param.pt")
            q.load_state_dict(params)

            s = env.reset(Parameters.datasetId)
            done = False
            score = 0.0
            while not done:
                a = q.sample_action(torch.from_numpy(s).float(), 0)
                s_prime, r, done = env.step(a)  # r은 그냥 0으로 설정
                inputs = np.concatenate([s, np.array([a])])
                r = cls.reward_model.r_hat(inputs)[0]
                done_mask = 0.0 if done else 1.0
                # data 생성할 때만 사용
                s = s_prime
                score += r
            Flow_time, machine_util, util, makespan, tardiness, lateness, t_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time, rtf = env.performance_measure()
            env.gantt_chart()
            print(util)



    @classmethod
    def learn_reward(cls, count):
        """
        Learn reward
        :return:
        """
        cls.load_reward_model(count)
        # labeled data 불러와서 train_reward 작업만 하자
        df = cls.reward_model.get_label()
        sa_t_1 = df.iloc[:, 0: (Hyperparameters.ds + Hyperparameters.da)].to_numpy().reshape(-1, Hyperparameters.size_sample_action, (Hyperparameters.input_layer + 1))
        sa_t_2 = df.iloc[:, (Hyperparameters.ds + Hyperparameters.da):(Hyperparameters.ds + Hyperparameters.da) * 2].to_numpy().reshape(-1, Hyperparameters.size_sample_action, (Hyperparameters.input_layer + 1))
        labels = df.iloc[:, (Hyperparameters.ds + Hyperparameters.da) * 2].to_numpy()[::Hyperparameters.size_sample_action]



        for epoch in range(Hyperparameters.reward_update):
            loss_avg = cls.reward_model.train_reward(sa_t_1, sa_t_2, labels)
            if loss_avg < 0.01:
                cls.save_reward_model(count)
                break
        cls.save_reward_model(count)
    @classmethod
    def normalize(cls, data):
        min_val = min(data)
        max_val = max(data)
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
        return normalized_data

    @classmethod
    def script_performance(cls, env, n_epi, epsilon, memory, score, type, makespan_list, util_list, score_list):
        Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, q_over_time, rtf = env.performance_measure()
        output_string = "--------------------------------------------------\n" + \
                        f"util : {util:.3f}\n" + \
                        f"n_episode: {n_epi}, score : {score:.1f}, eps : {epsilon * 100:.1f}%"
        print(output_string)
        if type:
            makespan_list.append(makespan)
            util_list.append(util)
            score_list.append(score)
        if Parameters.log_on:
            logging.info(f'performance :{output_string}')

        return makespan_list, util_list, score_list
    @classmethod
    def save_reward_model(cls, count):
        cls.reward_model.save(pathConfig.reward_model_params_path, count)

    @classmethod
    def load_reward_model(cls, count):
        if os.path.exists(f"{pathConfig.reward_model_params_path}{os.sep}{count}reward_model.pt"):
            cls.reward_model.load(pathConfig.reward_model_params_path, count)
    @classmethod
    def config_file_name(cls, name):
        cls.file_name = name