import pandas as pd
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from src.save_data.data_generator.compare_chart import *
from src.simulator.Simulator import *
import os
import sys



class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if Hyperparameters.reward_model_activation_function == "sigmoid":
            x = F.sigmoid(self.layer1(x))
            x = F.sigmoid(self.layer2(x))

        elif Hyperparameters.reward_model_activation_function == "tanh":
            x = torch.tanh(self.layer1(x))
            x = torch.tanh(self.layer2(x))

        elif Hyperparameters.reward_model_activation_function == "ReLU":
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))

        elif Hyperparameters.reward_model_activation_function == "leaky_ReLU":
            x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
            x = F.leaky_relu(self.layer2(x), negative_slope=0.01)

        elif Hyperparameters.reward_model_activation_function == "ELU":
            x = F.elu(self.layer1(x), alpha=1.0)
            x = F.elu(self.layer2(x), alpha=1.0)

        elif Hyperparameters.reward_model_activation_function == "SELU":
            x = F.selu(self.layer1(x))
            x = F.selu(self.layer2(x))

        elif Hyperparameters.reward_model_activation_function == "GELU":
            x = F.gelu(self.layer1(x))
            x = F.gelu(self.layer2(x))

        elif Hyperparameters.reward_model_activation_function == "Softplus":
            x = F.softplus(self.layer1(x))
            x = F.softplus(self.layer2(x))

        elif Hyperparameters.reward_model_activation_function == "Swish":
            x = self.layer1(x) * torch.sigmoid(self.layer1(x))
            x = self.layer2(x) * torch.sigmoid(self.layer2(x))
        x = self.layer3(x)  # 직접적인 선형 변환 결과를 반환
        return x

class RewardModel:
    def __init__(self, ds, da, lr=3e-4, size_sample_action=1,
                 max_size=2000, capacity=5e5):

        self.ds = ds  # 상태의 차원
        self.da = da  # 행동의 차원
        self.lr = lr  # 학습률
        self.opt = None  # 옵티마이저
        self.model = None  # 모델
        self.max_size = max_size  # 버퍼의 최대 크기
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

        self.count = 0

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
        for i in range(len(self.inputs)):
            df = pd.DataFrame(self.inputs[i])
            df.to_csv(f"{pathConfig.unlabeled_data_path}{os.sep}inputs{i}.csv", index=True)

    def r_hat_model(self, x):
        return self.model(torch.from_numpy(x).float())
    def r_hat(self, x):
        return self.r_hat_model(x).detach().numpy()

    def save(self, model_dir, count):
        torch.save(
            self.model.state_dict(), f'{model_dir}/{count}reward_model.pt'
        )

    def load(self, model_dir, count):
        self.model.load_state_dict(
            torch.load(f'{model_dir}/{count}reward_model.pt')
        )


    def compute_l2_loss(self, model):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for param in model.parameters():
            l2_loss = l2_loss + torch.norm(param, 2) ** 2
        return l2_loss

    def get_label(self):
        df = pd.read_csv(f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv", index_col=0)
        return df
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
        filtered_labels_len = 0
        # 총
        total_batch_index = np.random.permutation(max_len)

        num_epochs = int(np.ceil(max_len / self.train_batch_size))

        sum_loss = 0.0
        for epoch in range(num_epochs):
            # model parameter 초기화
            loss = 0.0
            # 마지막 인덱스를 계산하는 방법 (if epoch = 1 이면 128 * 2 이므로 256까지만 데이터를 다룸)
            last_index = (epoch + 1) * self.train_batch_size
            # total_batch_index의 크기를 벗어나면 안되니깐
            if last_index > max_len:
                last_index = max_len

            idxs = total_batch_index[epoch * self.train_batch_size:last_index]
            sa_t_1_t = sa_t_1[idxs]
            sa_t_2_t = sa_t_2[idxs]
            labels_t = labels[idxs]

            r_hat1 = self.r_hat_model(sa_t_1_t)
            r_hat2 = self.r_hat_model(sa_t_2_t)

            r_hat1 = r_hat1.sum(axis=1)
            r_hat2 = r_hat2.sum(axis=1)

            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
           #  r_hat_mean = abs(sum(r_hat1) / num_epochs)
            # cross entropy loss를 통해 실제 선호도 라벨 값과 뉴럴 넷에서 산출한 라벨 값을 비교함
            curr_loss = self.CEloss(r_hat, torch.tensor(labels_t).reshape(-1).long())
            sum_loss += curr_loss.item()
            loss += curr_loss # + r_hat_mean/ 10
            if Hyperparameters.parameter_regularization == True:
                l2_loss = self.compute_l2_loss(self.model)
                loss = curr_loss + 0.0001 * l2_loss
            # 현재 계산된 curr_loss 값을 int로 변환하여 저장

            # compute acc
            _, predicted = torch.max(r_hat.data, 1)
            labels_t = torch.tensor(labels_t).flatten()
            for idx in range(len(labels_t)):
                label = labels_t[idx]
                if label != 0.5:
                    filtered_labels_len += 1
                    if label == predicted[idx]:
                        correct += 1
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        accuracy = correct / filtered_labels_len
        print("reward value  ", r_hat1[0].item())
        print("loss  ", sum_loss / max_len)
        print("accuracy  ", accuracy)
        # return accuracy
        return sum_loss/max_len, accuracy
