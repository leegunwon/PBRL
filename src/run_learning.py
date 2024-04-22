from learner.Algorithm.PPO import PPO
from src.learner.Algorithm.DQN import *
from src.learner.Algorithm.DQN_action_masking import *
from src.learner.Algorithm.DQN_CNN import *
from src.learner.Algorithm.DDQN import *
from src.learner.Algorithm.PBRL import PBRL
from src.learner.common.Hyperparameters import *
from src.save_data.data_generator.compare_chart import *
from src.simulator.GanttChart import *
import plotly
import yaml
import logging

class Run_Simulator:

    def __init__(self):
        Parameters.set_time_to_string()  # 현재 시간 가져오는 코드 -> 로그 및 기록을 위함
        Parameters.set_absolute_path()

        Parameters.set_dataSetId(["sks_train_1"])  # 사용할 데이터셋 설정
        # Parameters.set_dataSetId(['sks_train_11'])

        with open(f'{pathConfig.absolute_path}{os.sep}hyperparameter.yaml', 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)

        Parameters.init_parameter_setting(config_data['engine'])
        Parameters.init_db_setting(config_data['database'])
        Parameters.set_plan_horizon(840)
        self.action_list = ["SPTSSU", "SSU", "EDD", "MST", "FIFO", "LIFO"]
        Simulator.init_simulator(Parameters.datasetId)  # 데이터셋 선택 후 데이터에 맞게 시뮬레이터 설정

        Hyperparameters.init_hyperparameter_setting(config_data['hyperparameter'])
        Hyperparameters.init_rl_config_setting(config_data['configRL'], self.action_list, Simulator)

        print("set complete")

    def main(self, mode, algorithm):
        logging.info(f"mode: {mode}")
        logging.info(f"algoritm: {algorithm}")
        if mode == "learning":
            if algorithm == 'dqn':
                if ActionManager.action_type == "action_masking":
                    DQN_Action_Masking.main()
                else:
                    DQN.main()
            elif algorithm == 'ddqn':
                DDQN.main()
            elif algorithm == 'dqn_cnn':
                DQN_CNN.main()
            elif algorithm == 'PPO':
                ppo = PPO()
                ppo.main()
            elif algorithm == 'PBRL':
                PBRL.main()
            elif algorithm == 'reward_model':
                # reward_model 학습 및 생성
                PBRL.learn_reward()

        elif mode == 'evaluate':
            if algorithm == "dqn":
                DQN.get_evaluate(f"{pathConfig.model_save_path}{os.sep}240209_233447", 100,
                                 ["sks_train_1"])
            elif algorithm == "PBRL":
                PBRL.evaluate()

        elif mode == "result":
            if algorithm == 'dqn':
                DQN.get_result(f"{pathConfig.model_save_path}{os.sep}240209_233447{os.sep}24param.pt", ["sks_train_1"])

        elif mode == "make_dataset":
            # 파라미터화 해야함
            # to_do_list
            # 기존의 counts를 하나씩 늘리고 20개 30개가 되면 저장하는 느낌
            # 기존 문제는 한 개의 트레젝토리가 너무 커져서 학습을 시킬 때 상벌이 애매함
            # 판단이 어려움
            # 새로운 트레젝토리
            # 전체 simulation 결과를 저장을 함
            # trajectory len을 하이퍼 파라미터로 설정하여 trajectory len 만큼의 action이 수행된 결과를 출력
            # 랜덤으로 시점을 설정하여 의사결정을 수행함.
            # 범위는 초반 dispatching을 수행한 후 부터 마지막 결정 - trajectory len 만큼
            Hyperparameters.mode = 1
            PBRL.main()

        elif mode == "labeling":
            app_run()


if True:
    simulator = Run_Simulator()
    # labeling, evaluate, learning, result
    # reward_model, dqn, PBRL
    simulator.main("learning", "dqn")  # dsp_rule = 개별 확인할 때만 사용하면 됨

# gantt chart 쑬 것인지
# 학습 방법, kpi목표
# 모든 디스패칭 룰 돌리기
