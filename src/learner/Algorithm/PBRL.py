import os

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging
import time
import plotly.graph_objects as go
from src.learner.common.ReplayBuffer import *
from src.learner.common.RewardModel import RewardModel
from src.simulator.Simulator import Simulator
from src.learner.common.Hyperparameters import *
from src.learner.common.Qnet import Qnet
from src.master_db.PBRL_DB_interface import save_episode_data_to_db
from src.master_db.PBRL_DB_interface import load_whole_labeled_data_from_db
from src.chart.ScatterPlot import *
from src.common.pathConfig import pathConfig

class PBRL:
    print("PBRL on")
    # reward model 바꿀 점 label 데이터를 바탕으로 학습 데이터 만들기
    reward_model = RewardModel(
        ds=Hyperparameters.input_layer,
        da=Hyperparameters.output_layer,
        lr=Hyperparameters.reward_lr,
        max_size=Hyperparameters.episode,
        size_sample_action=Hyperparameters.trajectory_segmentation_size)

    @classmethod
    def run_simulation_with_q_net_param(cls, ind, check_point):
        """
        주어진 체크포인트의 Q-네트워크를 사용하여 시뮬레이션을 실행하고 결과를 출력하는 메서드.

        Args:
            ind (int): 학습 세션의 인덱스. 체크포인트 경로에 사용됨.
            check_point (int): 로드할 체크포인트 번호.

        설명:
            - 지정된 체크포인트에서 Q-네트워크를 로드합니다.
            - 환경을 초기화하고, 에이전트가 행동을 선택하며 시뮬레이션을 진행합니다.
            - 시뮬레이션 결과(Flow time, Utilization 등)를 계산하고 간트 차트를 생성합니다.
        """
        # Q-네트워크 및 시뮬레이터 초기화
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        env = Simulator

        # 체크포인트에서 Q-네트워크 파라미터 로드
        try:
            checkpoint_path = os.path.join(pathConfig.reinforcement_model_params_path, str(ind),
                                           f"{check_point}q_net_param.pt")
            params = torch.load(checkpoint_path, weights_only=True)
            q.load_state_dict(params)
        except Exception as e:
            print(f"Failed to load checkpoint {check_point}: {e}")
            return  # 로드 실패 시 시뮬레이션 종료

        # 환경 초기화
        s = env.reset(Parameters.datasetId)
        done = False
        score = 0.0

        # 시뮬레이션 실행
        while not done:
            a = q.sample_action(s, epsilon=0)  # 탐험 없이 결정적 행동 선택
            s_prime, r, done = env.step(a)  # 환경에서 액션 실행 (보상 r은 0)
            s = s_prime  # 상태 업데이트
            score += r  # 총 점수 누적

        # 성능 측정
        Flow_time, machine_util, util, makespan, tardiness, lateness = env.performance_measure_for_pbrl()

        # 결과 출력 및 간트 차트 생성
        print(f"Simulation completed with score: {score}")
        print(f"Performance Metrics - Flow Time: {Flow_time}, Machine Utilization: {machine_util}, "
              f"Utilization: {util}, Makespan: {makespan}, Tardiness: {tardiness}, Lateness: {lateness}")
        env.gantt_chart()

    @classmethod
    def compute_l2_loss(self, model):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for param in model.parameters():
            l2_loss = l2_loss + torch.norm(param, 2) ** 2
        return l2_loss

    @classmethod
    def train(cls, q, q_target, memory, optimizer):
        """
        Q-네트워크를 학습하는 메서드.

        Args:
            q: 학습 중인 Q-네트워크.
            q_target: 타겟 Q-네트워크.
            memory: 경험 리플레이 버퍼.
            optimizer: Q-네트워크의 매개변수를 업데이트하는 옵티마이저.

        Returns:
            float: 학습 과정에서 발생한 총 손실의 값.
        """
        loss = 0  # 총 손실 초기화

        # 학습 루프 (10회 반복)
        for i in range(10):
            # 리플레이 버퍼에서 샘플 가져오기
            s, a, r, s_prime, done_mask = memory.sample(Hyperparameters.batch_size)

            # Q-값 계산 및 타겟 생성
            q_out = q(s)  # 현재 상태에서의 Q-값
            q_a = q_out.gather(1, a)  # 선택된 액션의 Q-값
            max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)  # 타겟 네트워크에서 최대 Q-값
            target = r + Hyperparameters.gamma * max_q_prime * done_mask  # Bellman 방정식에 의한 타겟 Q-값

            # 손실 계산
            curr_loss = F.smooth_l1_loss(q_a, target)

            # 역전파 및 네트워크 업데이트
            optimizer.zero_grad()
            loss += curr_loss  # 총 손실 누적
            curr_loss.backward()
            optimizer.step()

        return loss.item()  # 총 손실 반환

    @classmethod
    def main(cls, ind):
        """
        강화학습 에이전트의 주요 학습 루프.

        Args:
            ind (int): 현재 학습 세션의 인덱스 또는 식별자.
                         모델 로드 및 저장 시 사용됨.

        설명:
            이 메서드는 환경, Q-네트워크, 옵티마이저, 리플레이 버퍼를 초기화합니다.
            에피소드별로 환경을 실행하며, epsilon-greedy 방식으로 액션을 선택하고,
            보상을 예측 모델로 계산합니다. 전이 데이터를 리플레이 버퍼에 저장하고
            주기적으로 Q-네트워크를 업데이트합니다. 또한 성능 지표를 기록하며
            학습 과정에서 모델을 저장합니다.
        """
        print("main function run")

        # 환경 및 Q-네트워크 초기화
        env = Simulator
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target.load_state_dict(q.state_dict())
        optimizer = optim.Adam(q.parameters(), lr=Hyperparameters.learning_rate)
        memory = ReplayBuffer(Hyperparameters.buffer_limit)

        # 성능 지표를 추적하기 위한 리스트
        makespan_list = []
        score_list = []
        util_list = []
        loss_list = []
        loss = 0

        EPISODE_NUM = Hyperparameters.episode
        check_point = 0

        # 사전 학습된 보상 모델과 Q-net 로드
        cls.load_reward_model(ind)
        q = cls.load_q_net(q, ind)

        # 학습 루프
        for n_epi in range(EPISODE_NUM):
            # Epsilon 값 감소 (탐험-활용 균형 조정)
            epsilon = max(0.01, 0.8 - 0.9 / EPISODE_NUM * n_epi)

            # 새로운 에피소드를 위한 환경 초기화
            s = env.reset(Parameters.datasetId)
            done = False
            score = 0.0
            transition_lst = []

            while not done:
                # epsilon-greedy 방식으로 액션 선택
                a = q.sample_action(s, epsilon)

                # 환경에서 액션 실행
                s_prime, _, done = env.step(a)

                # 보상 예측을 위한 입력 데이터(s, a) 준비
                inputs = np.concatenate([s, np.array([a])])
                transition_lst.append(inputs)

                # 입력 데이터를 PyTorch 텐서로 변환
                inputs = torch.from_numpy(inputs)

                # 보상 모델로 보상 예측
                r = cls.reward_model.r_hat(inputs)[0]

                # 전이 데이터를 리플레이 버퍼에 저장
                done_mask = 0.0 if done else 1.0
                memory.put([s, a, r, s_prime, done_mask])
                s = s_prime
                score += r

                # 에피소드 종료 시 루프 종료
                if done:
                    break

            # 리플레이 버퍼에 충분한 샘플이 있을 경우 Q-네트워크 학습
            if memory.size() > 100:
                loss = cls.train(q, q_target, memory, optimizer)

            # 주기적으로 성능 기록 및 q_target 네트워크 업데이트
            if n_epi % 5 == 0:
                makespan_list, util_list, score_list = cls.script_performance_for_pbrl(
                    env, n_epi, epsilon, memory, score,
                    True, makespan_list, util_list, score_list
                )
                loss_list.append(loss)
                q_target.load_state_dict(q.state_dict())

            # 마지막 100개의 에피소드 동안 Q-net 파라미터 저장
            if n_epi >= EPISODE_NUM - 100:
                params = q.state_dict()
                file_name = f"{check_point}q_net_param.pt"
                file_path = os.path.join(f"{pathConfig.reinforcement_model_params_path}{os.sep}{ind}", file_name)
                torch.save(params, file_path)
                check_point += 1

            # 에피소드 transition 데이터를 데이터베이스에 저장
            save_episode_data_to_db(pd.DataFrame(transition_lst), n_epi)

        # PBRL 학습 곡선 차트 생성
        pbrl_Learning_scatter_plot(score_list, util_list, loss_list, ind)

    @classmethod
    def evaluate(cls, ind):
        """
        에이전트를 평가하여 최적의 체크포인트를 찾고, 해당 체크포인트에서 시뮬레이션을 실행하는 메서드.

        Args:
            ind (int): 학습 세션의 인덱스. 체크포인트와 모델 파일 경로에서 사용됨.

        설명:
            - 주어진 체크포인트에서 Q-네트워크를 로드하고, 시뮬레이터를 통해 에피소드를 실행합니다.
            - 각 체크포인트의 점수와 utilization을 평가합니다.
            - 최고 점수를 얻은 체크포인트와 최대 utilization을 얻은 체크포인트를 기록하고, 해당 체크포인트의 gantt chart를 출력합니다.
        """
        # 시뮬레이터와 Q-네트워크 초기화
        env = Simulator
        q = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target = Qnet(Hyperparameters.input_layer, Hyperparameters.output_layer)
        q_target.load_state_dict(q.state_dict())  # 타겟 네트워크 동기화

        # 평가 변수 초기화
        max_util = 0.0  # 최대 자원 활용도
        max_score = float('-inf')  # 초기 최대 점수를 매우 낮은 값으로 설정
        check_point_max_score, check_point_max_util = 0, 0  # 최고 점수 및 최대 활용도의 체크포인트 번호

        # 체크포인트 평가 루프
        for i in range(100):
            # 체크포인트 모델 파일 경로
            checkpoint_path = f"{pathConfig.reinforcement_model_params_path}{os.sep}{ind}{os.sep}{i}q_net_param.pt"

            # 체크포인트 파일이 존재할 경우 Q-네트워크 로드
            if os.path.exists(checkpoint_path):
                params = torch.load(checkpoint_path, weights_only=True)
                q.load_state_dict(params)

            # 시뮬레이터 초기화 및 에피소드 실행
            s = env.reset(Parameters.datasetId)
            done = False
            score = 0.0

            # 에피소드 루프
            while not done:
                a = q.sample_action(s, epsilon=0)  # 탐험 없이 결정적 행동 선택
                s_prime, r, done = env.step(a)  # 환경에서 액션 실행 (보상 r은 기본값 0)
                s = s_prime  # 상태 업데이트
                score += r  # 총 점수 누적

            # 환경 성능 측정
            Flow_time, util, makespan, tardiness, lateness = env.performance_measure_for_pbrl()

            # 최고 점수 체크포인트 업데이트
            if score > max_score:
                max_score = score
                check_point_max_score = i

            # 최대 활용도 체크포인트 업데이트
            if util > max_util:
                max_util = util
                check_point_max_util = i

        # 결과 출력
        print("check_point_max_score :", check_point_max_score)  # 최고 점수를 기록한 체크포인트
        print("max_score :", max_score)  # 최고 점수
        print("check_point_max_util :", check_point_max_util)  # 최대 활용도를 기록한 체크포인트
        print("max_utilization :", max_util)  # 최대 자원 활용도

        # 최고 점수와 최대 활용도 체크포인트에서 시뮬레이션 실행
        cls.run_simulation_with_q_net_param(ind, check_point_max_score)
        cls.run_simulation_with_q_net_param(ind, check_point_max_util)

    @classmethod
    def learn_reward(cls, ind):
        """
        보상 모델 학습 메서드.

        Args:
            ind (int): 현재 학습 세션의 인덱스. 보상 모델 저장에 사용됨.

        설명:
            - 데이터베이스에서 레이블된 데이터를 로드하여 전처리합니다.
            - 보상 모델을 학습시키고, 에포크별 손실 값을 기록합니다.
            - 손실 값을 그래프로 생성하여 저장합니다.
        """
        print("reward_learning_start")

        # 레이블 데이터 로드 및 전처리
        df = load_whole_labeled_data_from_db()
        ds, da = Hyperparameters.ds, Hyperparameters.da
        input_dim = Hyperparameters.input_layer + 1
        seg_size = Hyperparameters.trajectory_segmentation_size

        # sa_t_1과 sa_t_2 데이터 준비
        sa_t_1 = df.iloc[:, 0:(ds + da)].to_numpy().reshape(-1, seg_size, input_dim)
        sa_t_2 = df.iloc[:, (ds + da):(ds + da) * 2].to_numpy().reshape(-1, seg_size, input_dim)
        labels = df.iloc[:, (ds + da) * 2].to_numpy()[::seg_size]

        # 학습 루프
        loss_list = []
        for epoch in range(Hyperparameters.reward_update):
            average_loss, accuracy = cls.reward_model.train_reward(sa_t_1, sa_t_2, labels)
            loss_list.append(average_loss)

        # 손실 그래프 생성 및 저장
        cls._plot_and_save_loss_curve(loss_list, ind)

        # 보상 모델 저장
        cls.save_reward_model(ind)

        print("reward_learning_end")

    @staticmethod
    def _plot_and_save_loss_curve(loss_list, ind):
        """
        손실 값 그래프를 생성하고 HTML로 저장.

        Args:
            loss_list (list): 에포크별 평균 손실 값 리스트.
            ind (int): 세션 인덱스, 파일 저장에 사용됨.
        """
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(1, len(loss_list) + 1)), y=loss_list, mode='lines', name='Loss Curve'))
        fig.update_layout(title="Reward Model Loss Curve", xaxis_title="Epoch", yaxis_title="Loss")
        file_path = os.path.join(pathConfig.reward_model_learning_result_path, f"{ind}reward_loss.html")
        fig.write_html(file_path)
        fig.show()

        print(f"Reward Model Loss Curve saved at {file_path}")

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
        # print(output_string)
        if type:
            makespan_list.append(makespan)
            util_list.append(util)
            score_list.append(score)
        if Parameters.log_on:
            logging.info(f'performance :{output_string}')

        return makespan_list, util_list, score_list
    @classmethod
    def save_reward_model(cls, ind):
        print("reward estimator saved")
        cls.reward_model.save(pathConfig.reward_model_params_path, ind)

    @classmethod
    def load_reward_model(cls, ind):
        if os.path.exists(f"{pathConfig.reward_model_params_path}{os.sep}{ind}reward_model.pt"):
            print("reward estimator loaded")
            cls.reward_model.load(pathConfig.reward_model_params_path, ind)
    @classmethod
    def load_q_net(cls, q, ind):
        if os.path.exists(f"{pathConfig.reinforcement_model_params_path}{os.sep}{ind}reward_model.pt"):
            params = torch.load(f"{pathConfig.reinforcement_model_params_path}{os.sep}{ind}q_net_param.pt")
            q.load_state_dict(params)
        return q

    @classmethod
    def script_performance_for_pbrl(cls, env, n_epi, epsilon, memory, score, type, makespan_list, util_list, score_list):
        Flow_time, util, makespan, Tardiness_time, Lateness_time = env.performance_measure_for_pbrl()
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
    def config_file_name(cls, name):
        cls.file_name = name