# PBRL (Preference-Based Reinforcement Learning)

## 프로젝트 개요
PBRL은 **Preference-Based Reinforcement Learning(선호 기반 강화학습)**을 생산 공정 스케줄링에 적용한 프로젝트입니다. 
강화학습 환경에서 에이전트는 선호도 데이터를 기반으로 보상 모델을 학습하고, 이를 통해 최적의 행동 정책을 학습합니다.

---

## 주요 기능
- **시뮬레이션**: 설정한 규칙에 따라 생산 공정 스케줄링 시뮬레이션을 수행.
- **보상 모델 학습**: 사용자의 선호 데이터를 사용하여 보상 함수를 학습.
- **Query program**: 사용자의 선호도를 조사하는 프로그램.
- **Label generate**: 사용자의 선호도를 라벨을 자동으로 생성하는 프로그램.
- **강화학습 에이전트**: 보상 모델을 기반으로 환경에서 최적의 행동을 학습.
- **체크포인트 저장 및 로드**: 학습된 모델을 저장하고 필요한 시점에 재사용.
- **성능 평가**: 학습된 모델의 점수, 자원 활용도 등 주요 성능 지표를 측정.
- **시각화**: 간트 차트 및 학습 곡선을 통해 결과를 시각화.

---

## 설치 방법
1. 프로젝트 클론:
    ```bash
    git clone https://github.com/<your_username>/PBRL.git
    cd PBRL
    ```
2. 필요한 패키지 설치:
    ```bash
    pip install -r requirements.txt
    ```

---

## 사용 방법

### 1. 단순 시뮬레이션 동작
`run_simulator.py` 실행  
`run_simulator.py` 내부의 `simulator.main(mode, DSP_rule)`을 설정한 후 실행  

- **mode**:  
  - `dsp_run` : 선택한 `DSP_rule`을 기반으로 시뮬레이션을 수행  
  - `DSP_rule_check` : `src/common/Parameters.py`에 정의된 `DSP_rule_check` 파라미터 값이 `True`로 설정된 모든 `DSP_rule`로 시뮬레이션을 수행

- **DSP_rule** :
 사용 가능한 DSP_rule : [SPT, SPTSSU, SSU, LOR, MOR, EDD , MST, FIFO, LIFO] 중 하나 선택

### 2. 학습 시뮬레이션 동작
`run_learning.py` 파일 실행  
`run_learning.py` 내부의 `simulator.main(mode, algorithm, iteration)`을 설정한 후 실행  

- **mode**:
  - `learning` : 선택한 강화학습 알고리즘으로 에이전트를 학습
  - `evaluate` : 선택된 강화학습 알고리즘으로 학습된 에이전트의 학습 결과를 평가
  - `result` : 선택한 에이전트를 사용한 시뮬레이션 결과 확인
  - `query_program` : 사용자 선호도 조사 프로그램 실행
  - `label_generator` : 자동으로 선호도 조사하는 프로그램 실행 (현재 두 trajectory segmentation 중 setup time이 적은 segmentation을 선택하도록 구현)
  - `generate_reward_model` : 선호도 조사 데이터를 바탕으로 reward model을 학습

**algorithm** :
 사용 가능한 알고리즘 : [DQN, DDQN, DQN_action_masking, DQN_CNN, PBRL, PPO]
 검증된 알고리즘 [DQN, PBRL]

**iteration** : 
 실험을 반복해서 진행할 때 구분하기 위한 구분자
 ex) iteration=0 -> 생성되는 param 파일명 0param.pt


### 3. 데이터베이스 설정
현재 사용중인 DB 정보
```plaintext
db_user = 'root'
db_password = '1234'
db_host = 'localhost'
db_port = '3306'

sim_db_name = 'simulation_result' 
sim_table_name = 'episode_data'

label_db_name = 'query_result'
label_table_name = 'labeled_data'
```
- **sim_db** : 학습 시뮬레이션 도중 생성된 [state, action] 쌍 trajectory를 에피소드 별로 저장
- **label_db** : 사용자 선호도 조사 결과 데이터를 저장 
  - ex) 1번trajectory segmentation과 2번trajectory segmentation 선호도 조사 결과 데이터
  - [1번trajectory segmentation, 2번trajectory segmentation, label]


### 4. 시뮬레이션 결과 확인

```plaintext
simulator_result/
├── gantt_chart/                    # 간트 차트 결과 저장
├── reward_model_learning_result/   # 보상 모델 학습 결과
└── pbrl_learning_result/           # PBRL 학습 결과
    ├── matplotlib/                 # Matplotlib으로 시각화된 결과
    ├── seaborn/                    # Seaborn으로 시각화된 결과
    └── plotly/                     # Plotly로 시각화된 결과
```
                 


**gantt_chart** : evaluate mode 혹은 result mode 결과 저장
(파일명 : [시뮬레이션 실행 시간][활용된 데이터 셋]_gantt.html)

**reward_model_learning_result** : generate_reward_model mode 학습 지표 저장
(파일명 : [iteration]reward_loss.html)

**pbrl_learning_result** : learning mode 중 pbrl algorithm 학습 지표 저장
(파일명 : [iteration]Change_in_loss_[r_squared].html)

**4개의 학습 지표**
1. episode에 따른 cumulative reward 변화
2. episode에 따른 utilization 변화
3. cumulative reward와 utilization 간의 상관관계
4. episode에 따른 loss 변화
matplotlib, seaborn, plotly : 위 지표를 각각의 라이브러리를 활용하여 출력한 결과를 저장하는 곳

### 4. 파라미터 세팅
- **`src/Hyperparameter.yaml`**  
  자주 사용하는 파라미터 설정 파일.

- **`src/common/pathConfig.py`**  
  파일 경로를 관리하는 설정 파일.

- **`src/common/Parameters.py`**  
  시뮬레이터와 연관된 파라미터를 정의하는 파일.

- **`src/learner/common/Hyperparameters.py`**  
  학습 알고리즘과 연관된 파라미터를 정의하는 파일.

- **`src/chart/ChartParameters.py`**  
  PBRL 학습 지표 출력을 관리하는 파라미터 파일.

- **`src/params_data/reinforcement_learning`**  
  강화학습 에이전트 파라미터를 저장하는 디렉토리.

- **`src/params_data/reward_model_params`**  
  보상 함수 뉴럴넷 파라미터를 저장하는 디렉토리.
