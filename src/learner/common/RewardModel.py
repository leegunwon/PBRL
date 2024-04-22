import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from src.save_data.data_generator.compare_chart import *
from src.simulator.Simulator import *
import os
import sys
device = 'cpu'


def gen_net(in_size=1, out_size=1, H=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, H))
        net.append(nn.LeakyReLU())
        in_size = H
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation == 'sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())

    return net

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)  # 직접적인 선형 변환 결과를 반환
        return x



class RewardModel:
    def __init__(self, ds, da,
                 ensemble_size=3, lr=3e-4, size_sample_action=1,
                 max_size=1000, activation='tanh', capacity=5e5):

        self.ds = ds  # 상태의 차원
        self.da = da  # 행동의 차원
        self.de = ensemble_size  # 앙상블의 크기
        self.lr = lr  # 학습률
        self.ensemble = []  # 모델 앙상블
        self.paramlst = []  # 앙상블 멤버들의 파라미터 목록
        self.opt = None  # 옵티마이저
        self.model = None  # 모델
        self.max_size = max_size  # 버퍼의 최대 크기
        self.activation = activation  # 활성화 함수
        self.size_sample_action = size_sample_action  # 세그먼트의 크기

        self.capacity = int(capacity)  # 버퍼의 용량
        self.buffer_seg1 = np.empty((self.capacity, size_sample_action, self.ds + self.da),
                                    dtype=np.float32)  # 첫 번째 세그먼트 버퍼
        self.buffer_seg2 = np.empty((self.capacity, size_sample_action, self.ds + self.da),
                                    dtype=np.float32)  # 두 번째 세그먼트 버퍼
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)  # 레이블 버퍼
        self.buffer_index = 0  # 버퍼 인덱스
        self.buffer_full = False  # 버퍼가 가득 찼는지 여부

        self.construct_model()  # 앙상블 생성
        self.inputs = []  # 입력 데이터 리스트
        self.train_batch_size = 128  # 학습 미니배치 크기
        self.CEloss = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수

        self.count = 0 #

    def construct_model(self):
        """
        앙상블 뉴럴넷 구조를 사용
        de : 사용할 앙상블 뉴럴넷 갯수
        """

        self.model = SimpleNN(input_size=self.ds + self.da, hidden_size=256, output_size=1)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def add_data(self, obs, act, done):
        """
        강화학습 도중 발생한 데이터를 저장하는 공간
        FIFO의 형태로 데이터를 관리
        inputs의 경우 : 입력 데이터 (상태-액션 쌍) 에피소드 단위로 저장됨
        """
        sa_t = np.concatenate([obs, np.array([act])], axis=-1)

        # flattening 해서 상태-액션 쌍을 입력으로 활용
        flat_input = sa_t.reshape(1, self.da + self.ds)

        # 입력데이터를 저장하는 리스트가 비어있으면 처음 시작으로 간주
        init_data = len(self.inputs) == 0

        if init_data:
            # 입력 데이터를 저장해간다.
            self.inputs.append(flat_input)
        # 에피소드가 끝났을 때
        elif done:
            # 같은 차원으로 합침
            # ex) inputs[-1] = [1, 2, 3] flat_input = [4, 5, 6] -> [1, 2, 3, 4, 5, 6]
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            # FIFO
            # 아래의 과정을 통해 맨 앞 데이터를 지우고 가장 뒤의 데이터를 빈 리스트로 생성함.
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
            # 끝났다는 표시?
            self.inputs.append([])
        else:
            # 만약 마지막 입력 데이터 저장소 값이 빈 리스트라면 -> 현재의 input 값으로 대체한다
            if len(self.inputs[-1]) == 0:
                self.inputs[-1] = flat_input
            # 만약 마지막 입력 데이터 저장소 값이 빈 리스트가 아니면 -> 현재의 input 값 추가
            else:
                # 계속해서 상태-액션 쌍을 이어붙여감 inputs 리스트 한 셀에 trajectory를 담는 과정 (범위는 episode)
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])

    def data_save(self):
        for i in range(len(self.inputs)-1000, len(self.inputs)):
            df = pd.DataFrame(self.inputs[i])
            df.to_csv(f"{pathConfig.unlabeled_data_path}{os.sep}inputs{i- len(self.inputs)}.csv", index=True)

    def r_hat_model(self, x):
        # the network parameterizes r hat in eqn 1 from the paper
        return self.model(torch.from_numpy(x).float().to(device))


    def r_hat(self, x):
        # they say they average the rewards from each member of the ensemble, but I think this only makes sense if the rewards are already normalized
        # but I don't understand how the normalization should be happening right now :(
        r_hats = []
        r_hats.append(self.r_hat_model(x).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        # 이거 왜 평균 구함?

        return r_hats

    def save(self, model_dir):
        torch.save(
            self.model.state_dict(), '%s/reward_model.pt' % (model_dir)
        )

    def load(self, model_dir):
        self.model.load_state_dict(
            torch.load('%s/reward_model.pt' % (model_dir))
        )

    def get_train_acc(self):
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = np.random.permutation(max_len)
        batch_size = 256
        num_epochs = int(np.ceil(max_len / batch_size))

        total = 0
        for epoch in range(num_epochs):
            last_index = (epoch + 1) * batch_size
            if (epoch + 1) * batch_size > max_len:
                last_index = max_len

            sa_t_1 = self.buffer_seg1[epoch * batch_size:last_index]
            sa_t_2 = self.buffer_seg2[epoch * batch_size:last_index]
            labels = self.buffer_label[epoch * batch_size:last_index]
            labels = torch.from_numpy(labels.flatten()).long().to(device)
            total += labels.size(0)
            # get logits
            r_hat1 = self.r_hat_model(sa_t_1)
            r_hat2 = self.r_hat_model(sa_t_2)
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
            _, predicted = torch.max(r_hat.data, 1)
            correct = (predicted == labels).sum().item()

        return correct

    def put_queries(self, sa_t_1, sa_t_2, labels):
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index


    def get_label(self):
        df = pd.read_csv(f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv", index_col=0)
        return df
        # return sa_t_1, sa_t_2, labels

    def uniform_sampling(self):
        """
        mode 2에서 활용
        생성한 labeled data를 불러옴
        sa_t_1과 sa_t_2 label에 불러온 데이터를 저장해줌
        put_query 함수를 통해서 sa_t_1, sa_t_2, label을 buffer로 다시 저장함
        buffer train 단계에서 buffer에 저장된 데이터로 학습을 수행함
        :return:
        """
        df = pd.read_csv(f"labeled_data", index_col=0)

        # get labels
        # sa_t_1, sa_t_2, labels = self.get_label(sa_t_1, sa_t_2)

        # if len(labels) > 0:
            # self.put_queries(sa_t_1, sa_t_2, labels)
        # return len(labels)
        return 10
    def train_reward(self, sa_t_1, sa_t_2, labels):
        """
        train reward model
        사용 방법 seg1 buffer와 seg2 buffer를 채워둔 후
        두 buffer에서 seg들을 추출해서 라벨 값과 현재 뉴럴넷이 산출하는 라벨 값을 비교하여 loss를 계산한다.
        """
        # ensemble의 loss들을 저장한 모양

        # max_len : buffer가 채워져 있는 만큼
        correct = 0
        max_len = len(labels)
        # 총
        total_batch_index = np.random.permutation(max_len)
            # 0 ~ max_len -1 사이의 리스트를 [0, 1, ... max_len-1] 랜덤으로 섞음.
            # 각 앙상블 멤버가 규칙적인 데이터 순서를 학습하지 않도록 랜덤으로 데이터를 학습하도록 함.
        # 소숫점을 올림함.
        # 전체 데이터를 학습하기 위해 몇번 반복해야할 지 계산
        num_epochs = int(np.ceil(max_len / self.train_batch_size))

        for epoch in range(num_epochs):
            # model parameter 초기화
            self.opt.zero_grad()
            loss = 0.0
            # 마지막 인덱스를 계산하는 방법 (if epoch = 1 이면 128 * 2 이므로 256까지만 데이터를 다룸)
            last_index = (epoch + 1) * self.train_batch_size
            # total_batch_index의 크기를 벗어나면 안되니깐
            if last_index > max_len:
                last_index = max_len

                # get random batch
                # total_batch_index에서 랜덤한 index를 추출한다.
            idxs = total_batch_index[epoch * self.train_batch_size:last_index]
            sa_t_1_t = sa_t_1[idxs]
            sa_t_2_t = sa_t_2[idxs]
            labels_t = labels[idxs]

            # get logits
            # 각각의 상태-액션 쌍의 보상 값을 계산함 (앙상블의 보상 값들을 같이 구해서 평균을 구함)
            r_hat1 = self.r_hat_model(sa_t_1_t)
            r_hat2 = self.r_hat_model(sa_t_2_t)
            # axis=1 이므로 행을 기준으로 합 계산.
            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)
            # 각각의 요소 별로 더함
            #  a: tensor([[1, 2],  b : tensor([[5, 6],  cat([a, b] axis=-1 tensor([[1, 2, 5, 6],
            #            [3, 4]])             [7, 8]])                            [3, 4, 7, 8]])
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
            # cross entropy loss를 통해 실제 선호도 라벨 값과 뉴럴 넷에서 산출한 라벨 값을 비교함
            curr_loss = self.CEloss(r_hat, torch.tensor(labels_t).reshape(-1).long())
            loss += curr_loss
            # 현재 계산된 curr_loss 값을 int로 변환하여 저장
            curr_loss.item()
            # compute acc
            _, predicted = torch.max(r_hat.data, 1)
            correct += (predicted.reshape(-1, 1) == torch.tensor(labels_t)).sum().item()
            accuracy = correct / len(labels)

            loss.backward()
            self.opt.step()

        return accuracy


def reward_dateset_generator(trajectory_length):
    """
    랜덤 룰로 시뮬레이션을 돌림
    추가해야 될 부분
    random_rule 함수 내에 trajectory 단위로 저장하는 방법을 추가해야 됌
    db를 사용해야 되나?
    이 gantt_history_df도 저장해야 됨.
    그리고 df를 바탕으로 점수 산출하는 알고리즘도 제작 ( df 순서를 바탕으로 어떻게 점수 산출하지? 기존 bradley-Terry 모델 분석)
    현재 20씩 저장하는 방식은 곤란할지도
    현재 결과는 너무 랜덤해서 리워드가 개판임


    :return:
    """
    Simulator.run_random_rule(trajectory_length)
