import torch.nn.functional as F
import torch.nn as nn
from src.simulator.Simulator import *
import torch
import numpy as np
import os


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if Hyperparameters.reward_model_activation_function == "sigmoid":
            x = F.sigmoid(self.layer1(x))
            x = F.sigmoid(self.layer2(x))
            x = F.sigmoid(self.layer3(x))

        elif Hyperparameters.reward_model_activation_function == "tanh":
            x = torch.tanh(self.layer1(x))
            x = torch.tanh(self.layer2(x))
            x = torch.tanh(self.layer3(x))

        elif Hyperparameters.reward_model_activation_function == "ReLU":
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))

        elif Hyperparameters.reward_model_activation_function == "leaky_ReLU":
            x = F.leaky_relu(self.layer1(x), negative_slope=0.01)
            x = F.leaky_relu(self.layer2(x), negative_slope=0.01)
            x = F.leaky_relu(self.layer3(x), negative_slope=0.01)

        elif Hyperparameters.reward_model_activation_function == "ELU":
            x = F.elu(self.layer1(x), alpha=1.0)
            x = F.elu(self.layer2(x), alpha=1.0)
            x = F.elu(self.layer3(x), alpha=1.0)

        elif Hyperparameters.reward_model_activation_function == "SELU":
            x = F.selu(self.layer1(x))
            x = F.selu(self.layer2(x))
            x = F.selu(self.layer3(x))

        elif Hyperparameters.reward_model_activation_function == "GELU":
            x = F.gelu(self.layer1(x))
            x = F.gelu(self.layer2(x))
            x = F.gelu(self.layer3(x))

        elif Hyperparameters.reward_model_activation_function == "Softplus":
            x = F.softplus(self.layer1(x))
            x = F.softplus(self.layer2(x))
            x = F.softplus(self.layer3(x))

        elif Hyperparameters.reward_model_activation_function == "Swish":
            x = self.layer1(x) * torch.sigmoid(self.layer1(x))
            x = self.layer2(x) * torch.sigmoid(self.layer2(x))
            x = self.layer3(x) * torch.sigmoid(self.layer3(x))
        x = self.layer4(x)  # 직접적인 선형 변환 결과를 반환
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
        self.train_batch_size = 64  # 학습 미니배치 크기
        self.CEloss = nn.CrossEntropyLoss()  # 교차 엔트로피 손실 함수

        self.count = 0

    def construct_model(self):
        """
        앙상블 뉴럴넷 구조를 사용
        de : 사용할 앙상블 뉴럴넷 갯수
        """
        self.model = SimpleNN(input_size=self.ds + self.da, hidden_size=64, output_size=1)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)


    def r_hat_model(self, x):
        return self.model(x.float())
    def r_hat(self, x):
        return self.r_hat_model(x).detach().numpy()

    def save(self, model_dir, count):
        torch.save(
            self.model.state_dict(), f'{model_dir}/{count}reward_model.pt'
        )

    def load(self, model_dir, count):
        self.model.load_state_dict(
            torch.load(f'{model_dir}/{count}reward_model.pt', weights_only=True)
        )


    def compute_l2_loss(self, model):
        l2_loss = torch.tensor(0.0, requires_grad=True)
        for param in model.parameters():
            l2_loss = l2_loss + torch.norm(param, 2) ** 2
        return l2_loss





def train_reward(self, sa_t_1, sa_t_2, labels):
    """
    보상 모델을 학습하는 메서드.

    Args:
        sa_t_1 (ndarray): 첫 번째 세그먼트의 상태-액션 데이터.
        sa_t_2 (ndarray): 두 번째 세그먼트의 상태-액션 데이터.
        labels (ndarray): 두 세그먼트에 대한 비교 라벨 (0, 0.5, 1).

    Returns:
        tuple: (평균 손실, 정확도)
    """
    # 장치 설정 (GPU가 가능하면 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 데이터 및 초기 변수 설정
    max_len = len(labels)
    num_batches = int(np.ceil(max_len / self.train_batch_size))
    labels_tensor = torch.tensor(labels).reshape(-1).long().to(device)
    total_indices = np.random.permutation(max_len)  # 데이터 인덱스 섞기

    total_loss = 0.0
    total_correct = 0
    total_filtered_labels = 0

    # 배치별 학습 루프
    for batch_idx in range(num_batches):
        # 배치 데이터 준비
        start_idx = batch_idx * self.train_batch_size
        end_idx = min((batch_idx + 1) * self.train_batch_size, max_len)
        batch_indices = total_indices[start_idx:end_idx]

        sa_t_1_batch = torch.tensor(sa_t_1[batch_indices]).to(device)
        sa_t_2_batch = torch.tensor(sa_t_2[batch_indices]).to(device)
        labels_batch = labels_tensor[batch_indices]

        # 모델 예측
        r_hat1 = self.r_hat_model(sa_t_1_batch).sum(dim=1)
        r_hat2 = self.r_hat_model(sa_t_2_batch).sum(dim=1)

        # 로지스틱 확률 계산
        s = torch.sigmoid(r_hat1 - r_hat2)
        logistic_prob = torch.cat([s, 1 - s], dim=-1)

        # 손실 계산
        batch_loss = self.CEloss(logistic_prob, labels_batch)

        # L2 정규화 추가
        if Hyperparameters.parameter_regularization:
            l2_loss = self.compute_l2_loss(self.model)
            batch_loss += 0.0001 * l2_loss

        # 역전파 및 최적화
        self.opt.zero_grad()
        batch_loss.backward()
        self.opt.step()

        # 배치 손실 업데이트
        total_loss += batch_loss.item()

        # 정확도 계산
        with torch.no_grad():
            _, predicted = torch.max(logistic_prob, dim=1)
            valid_mask = labels_batch != 0.5  # 라벨 값이 0.5가 아닌 데이터 필터링
            filtered_labels = labels_batch[valid_mask]
            filtered_predictions = predicted[valid_mask]
            total_correct += (filtered_predictions == filtered_labels).sum().item()
            total_filtered_labels += valid_mask.sum().item()

    # 평균 손실 및 정확도 계산
    average_loss = total_loss / max_len
    accuracy = total_correct / total_filtered_labels if total_filtered_labels > 0 else 0.0

    return average_loss, accuracy
