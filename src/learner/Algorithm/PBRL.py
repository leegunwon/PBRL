import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

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
        ensemble_size=Hyperparameters.ensemble_size,
        lr=Hyperparameters.reward_lr,
        max_size=Hyperparameters.max_size,
        size_sample_action=Hyperparameters.size_sample_action,
        activation=Hyperparameters.activation)

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
    def main(cls):
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
        score = 0.0
        print_interval = 20
        optimizer = optim.Adam(q.parameters(), lr=Hyperparameters.learning_rate)

        save_directory = f"{pathConfig.reward_model_params_path}"
        # model load
        if os.path.exists(f"{save_directory}{os.sep}q_net_param.pt"):
            params = torch.load(f"{save_directory}{os.sep}q_net_param.pt")
            q.load_state_dict(params)

        for n_epi in range(Hyperparameters.episode):
            epsilon = max(0.01, 0.8 - 0.001 * n_epi)
            s = env.reset(Parameters.datasetId)
            done = False
            while not done:
                if (Hyperparameters.mode == 1):
                    a = random.randint(0, len(Hyperparameters.action_list) - 1)
                else:
                    a = q.sample_action(torch.from_numpy(s).float(), epsilon)

                s_prime, r, done = env.step(a) # r은 그냥 0으로 설정
                inputs = np.concatenate([s, np.array([a])])
                r = cls.reward_model.r_hat(inputs)[0][0]
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

            if n_epi % print_interval == 0 and n_epi != 0:
                print("# of episode :{}, avg score : {:.1f} eps : {:.3f}".format(n_epi, score / print_interval, epsilon))
                score = 0.0

        # 결과 및 파라미터 저장
        params = q.state_dict()
        file_name = "q_net_param.pt"
        file_path = os.path.join(save_directory, file_name)
        torch.save(params, file_path)
        if (Hyperparameters.mode == 1 or Hyperparameters.mode == 4):
            cls.reward_model.data_save()

        print("학습이 종료되었습니다")

    @classmethod
    def evaluate(cls):
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
        score = 0.0
        save_directory = f"{pathConfig.reward_model_params_path}"
        # model load
        if os.path.exists(f"{save_directory}{os.sep}q_net_param.pt"):
            params = torch.load(f"{save_directory}{os.sep}q_net_param.pt")
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
        env.gantt_chart()
        Flow_time, machine_util, util, makespan, tardiness, lateness, t_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time, rtf = env.performance_measure()


    @classmethod
    def learn_reward(cls):
        """
        Learn reward
        :return:
        """
        cls.load_reward_model()
        # labeled data 불러와서 train_reward 작업만 하자
        df = cls.reward_model.get_label()
        sa_t_1 = df.iloc[:, 0:23].to_numpy().reshape(-1, Hyperparameters.size_sample_action, (Hyperparameters.da + Hyperparameters.ds))
        sa_t_2 = df.iloc[:, 23:46].to_numpy().reshape(-1, Hyperparameters.size_sample_action, (Hyperparameters.da + Hyperparameters.ds))
        labels = df.iloc[:, 46].to_numpy()[::3]
        # buffer에 넣어줘야 함.

        for epoch in range(Hyperparameters.reward_update):
            train_acc = cls.reward_model.train_reward(sa_t_1, sa_t_2, labels)
            print(train_acc)
            if train_acc > 0.97:
                break
        cls.save_reward_model()

        print("Reward function is updated!! ACC: " + str(train_acc))

    @classmethod
    def script_performance(cls, env, n_epi, epsilon, memory, score, type, makespan_list, q_over_time_list, score_list):
        Flow_time, machine_util, util, makespan, Tardiness_time, Lateness_time, T_max, q_time_true, q_time_false, q_job_t, q_job_f, q_over_time, rtf = env.performance_measure()

        output_string = "--------------------------------------------------\n" + \
                        f"flow time: {Flow_time}, util : {util:.3f}, makespan : {makespan}, rtf: {rtf}\n" + \
                        f"Tardiness: {Tardiness_time}, Lateness : {Lateness_time}, T_max : {T_max}\n" + \
                        f"q_true_op: {q_time_true}, q_false_op : {q_time_false}, q_true_job : {q_job_t}, q_false_job : {q_job_f}, q_over_time : {q_over_time}\n" + \
                        f"n_episode: {n_epi}, score : {score:.1f}, n_buffer : {memory.size()}, eps : {epsilon * 100:.1f}%"
        print(output_string)
        if type:
            makespan_list.append(makespan)
            q_over_time_list.append(q_over_time)
            score_list.append(score)
        if Parameters.log_on:
            logging.info(f'performance :{output_string}')

        return makespan_list, q_over_time_list, score_list
    @classmethod
    def save_reward_model(cls):
        cls.reward_model.save(pathConfig.reward_model_params_path)

    @classmethod
    def load_reward_model(cls):
        if os.path.exists(f"{pathConfig.reward_model_params_path}{os.sep}reward_model.pt"):
            cls.reward_model.load(pathConfig.reward_model_params_path)
