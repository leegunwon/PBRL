import os

from src.learner.common.Qnet import *
from src.learner.common.ReplayBuffer import *
from src.learner.common.Hyperparameters import *
import torch.nn.functional as F
import torch.optim as optim
from src.simulator.Simulator import *
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import logging
from src.common.Parameters import *

class DQN:
    print("DQN on")

    @classmethod
    def train(cls, q, q_target, memory, optimizer):
        for i in range(10):
            s, a, r, s_prime, done_mask = memory.sample(Hyperparameters.batch_size)
            # q.number_of_time_list[a] += 1
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
        Hyperparameters.mode : 1, 2, 3, 4
        1 : random하게 시뮬레이션을 수행하고 그 결과를 memory에 저장
        2 :
        """
        print("main function run")
        env = Simulator
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target.load_state_dict(q.state_dict())
        memory = ReplayBuffer(Hyperparameters.buffer_limit)
        makespan_list = []
        score_list = []
        util_list = []
        loss_list = []
        loss = 0
        optimizer = optim.Adam(q.parameters(), lr=Hyperparameters.learning_rate)
        save_directory = f"{pathConfig.reinforcement_model_params_path}"
        check_point = 0
        # model load
        cls.load_reward_model(count)
        cls.load_q_net(q, count)


        # cls.learn_reward()
        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0.01, 0.8 - 0.9 /Hyperparameters.episode * n_epi)
            s = env.reset(Parameters.datasetId)
            epsilon_test = 0
            done = False
            score = 0.0
            score_sub = 0.0
            while not done:
                # 뭐가 바뀌는 걸까? 아마도 q value가 바뀜 즉 action을
                a = q.sample_action(s, epsilon)
                s_prime, r, done = env.step(a)

                done_mask = 0.0 if done else 1.0
                memory.put([s, a, r, s_prime, done_mask])
                s = s_prime
                score += r

                if done:
                    break

            if (memory.size() > 100):
                loss = cls.train(q, q_target, memory, optimizer)

            if (n_epi % 5==0):
                makespan_list, util_list, score_list = cls.script_performance(env, n_epi, epsilon, memory, score,
                                                                              True, makespan_list, util_list,
                                                                              score_list)
                loss_list.append(loss)
                q_target.load_state_dict(q.state_dict())

            if n_epi >= Hyperparameters.episode * 0.9:
                params = q.state_dict()
                file_name = f"{check_point}q_net_param.pt"
                file_path = os.path.join(f"{pathConfig.reinforcement_model_params_path}{os.sep}{count}", file_name)
                torch.save(params, file_path)
                check_point += 1

        max_score = max(score_list)
        score_list = [s/max_score for s in score_list]

        fig = make_subplots(rows=1, cols=2)
        df = pd.DataFrame({'x': list(range(1, len(score_list)+1)), 'score': score_list})
        fig1 = px.scatter(df, x='x', y='score', trendline='ols', title='Score')
        df = pd.DataFrame({'score': score_list, 'util': util_list})
        fig2 = px.scatter(df, x='score', y='util', trendline='ols', title='Score vs Util Relationship')
        for trace in fig1.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig2.data:
            fig.add_trace(trace, row=1, col=2)
        fig.show()
        print("학습이 종료되었습니다")

    @classmethod
    def get_result(cls, parameter, dataSets):
        env = Simulator
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        memory = ReplayBuffer(Hyperparameters.buffer_limit)
        params = torch.load(parameter)
        q.load_state_dict(params)
        q.eval()
        for data_id in dataSets:
            s = env.reset(data_id)
            done = False
            score = 0.0
            while not done:
                epsilon = 1
                a, a_list = q.select_action(torch.from_numpy(s).float(), epsilon)
                s_prime, r, done = env.step(a)
                s = s_prime
                score += r
                if done:
                    break
            Flow_time, machine_util, util, makespan, tardiness, lateness, t_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time, rtf = env.performance_measure()
            env.gantt_chart()
            print(f"dataset: {data_id}")
            print("util:", util)
            print("Tardiness:", tardiness)
            if Parameters.log_on:
                logging.info(f"dataset: {data_id}")
                logging.info(f"util: {util}")
            print("평가가 종료되었습니다.")

    @classmethod
    def get_evaluate(cls, checkpoint_path, number_of_checkpoint, datasets):
        env = Simulator
        save_directory = f"{pathConfig.model_save_path}"
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        # model load
        if os.path.exists(f"{save_directory}{os.sep}q_net_param.pt"):
            params = torch.load(f"{save_directory}{os.sep}q_net`_`param.pt")
            q.load_state_dict(params)
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        mean_reward_by_checkpoint = {}
        max_reward_by_checkpoint = {}
        for check_point_number in range(100):
            check_point = f"{checkpoint_path}/{check_point_number}param.pt"
        params = torch.load(check_point)
        q.load_state_dict(params)
        q.eval()
        reward_list = []
        for dataset in datasets:
            s = env.reset(dataset)
            done = False
            score = 0.0
            while not done:
                epsilon = 1
                a, a_list = q.select_action(torch.from_numpy(s).float(), epsilon)
                # print(a_list)
                # print(a)
                s_prime, r, done = env.step(a)
                # print(r)
                s = s_prime
                score += r
                if done:
                    break
            Flow_time, machine_util, util, makespan, tardiness, lateness, t_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time, rtf = env.performance_measure()
            reward_list.append(score)
            if Parameters.log_on:
                logging.info(f"checkpoint: {check_point_number}")
                logging.info(f"dataset: {dataset}")
                logging.info(f"score:{score}")
            print(f"checkpoint: {check_point_number}")
            print(f"dataset: {dataset}")
            print(f"score:{score}")
        mean_reward_by_checkpoint[check_point_number] = sum(reward_list) / len(reward_list)
        max_reward_by_checkpoint[check_point_number] = max(reward_list)

        max_check_point = max(max_reward_by_checkpoint.items(), key=lambda x: x[1])[0]
        mean_check_point = max(mean_reward_by_checkpoint.items(), key=lambda x: x[1])[0]

        print("평가가 종료되었습니다.")

    @classmethod
    def script_performance(cls, env, n_epi, epsilon, memory, score, type, makespan_list, util_list, score_list):
        Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, q_over_time, rtf = env.performance_measure()

        output_string = "--------------------------------------------------\n" + \
                        f"flow time: {Flow_time}, util : {util:.3f}, makespan : {makespan}, rtf: {rtf}\n" + \
                        f"Tardiness: {Tardiness_time}, Lateness : {Lateness_time}, T_max : {T_max}\n" + \
                        f"q_true_op: {q_time_true}, q_false_op : {q_time_false}, q_true_job : {q_job_t}, q_false_job : {q_job_f}, q_over_time : {q_over_time}\n" + \
                        f"n_episode: {n_epi}, score : {score:.1f}, n_buffer : {memory.size()}, eps : {epsilon * 100:.1f}%"
        print(output_string)
        if type:
            makespan_list.append(makespan)
            util_list.append(q_over_time)
            score_list.append(score)
        if Parameters.log_on:
            logging.info(f'performance :{output_string}')
        return makespan_list, util_list, score_list
