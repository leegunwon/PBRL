from learner.Algorithm.PPO import PPO
from src.learner.Algorithm.DQN import *
from src.learner.Algorithm.DQN_action_masking import *
from src.learner.Algorithm.DQN_CNN import *
from src.learner.Algorithm.DDQN import *
from src.learner.Algorithm.PBRL import PBRL
from src.save_data.data_generator.compare_chart import *
from src.save_data.data_generator.data_generator import *
import plotly.graph_objects as go
import yaml
import logging
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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

    def main(self, mode, algorithm, iteration):
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
                r_sqrd = []
                for rep in range(iteration):
                    r_sqrd.append(PBRL.main(rep))
                print(r_sqrd)

            elif algorithm == 'reward_model':
                # reward_model 학습 및 생성
                for j in range(iteration):
                    PBRL.learn_reward(j)
                    Hyperparameters.reward_update = 1500

        elif mode == 'evaluate':
            if algorithm == "dqn":
                DQN.get_evaluate(f"{pathConfig.model_save_path}{os.sep}240209_233447", 100,
                                 ["sks_train_1"])
            elif algorithm == "PBRL":
                for repp in range(iteration):
                    PBRL.evaluate(repp)

        elif mode == "result":
            if algorithm == 'dqn':
                DQN.get_result(f"{pathConfig.model_save_path}{os.sep}240209_233447{os.sep}24param.pt", ["sks_train_1"])

        elif mode == "make_dataset":
            Hyperparameters.mode = 1
            for repp in range(iteration):
                PBRL.main(-1)
                generate_label()

        elif mode == "query_program":
            app_run()

        elif mode == "label_generator":
            generate_label()


if True:
    simulator = Run_Simulator()
    # mode : query_program, evaluate, learning, result, make_dataset, label_generator
    # algorithm : reward_model, dqn, PBRL
    iteration_count = 1
    # simulator.main(mode="make_dataset", algorithm="PBRL", iteration=iteration_count)
    simulator.main(mode="learning", algorithm="reward_model", iteration=iteration_count)
    simulator.main(mode="learning", algorithm="PBRL", iteration=iteration_count)
    # util_sum = 0
    # for i in range(iteration_count):
    #     util = PBRL.evaluate(i)
    #     util_sum += util
    # print(util_sum)


# gantt chart 쑬 것인지
# 학습 방법, kpi목표
# 모든 디스패칭 룰 돌리기
