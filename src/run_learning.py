from src.learner.Algorithm.PPO import PPO
from src.learner.Algorithm.DQN import *
from src.learner.Algorithm.DQN_action_masking import *
from src.learner.Algorithm.DQN_CNN import *
from src.learner.Algorithm.DDQN import *
from src.learner.Algorithm.PBRL import PBRL
from src.simulator.data_generator.compare_chart import *
from src.simulator.data_generator.data_generator import *
from src.master_db.PBRL_DB_interface import delete_episode_data_from_db, get_simulation_result_record_count
from src.master_db.PBRL_DB_interface import delete_labeled_data_from_db, get_labeled_data_record_count
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
        Hyperparameters.init_reward_model_setting(config_data['config_reward_model'])

        print("set complete")

    def main(self, mode, algorithm="", iteration=-1):
        logging.info(f"mode: {mode}")
        logging.info(f"algoritm: {algorithm}")
        if mode == "learning":
            if algorithm == 'dqn':
                if ActionManager.action_type == "action_masking":
                    DQN_Action_Masking.main()
                else:
                    PBRL.main(iteration)
            elif algorithm == 'ddqn':
                DDQN.main()
            elif algorithm == 'dqn_cnn':
                DQN_CNN.main()
            elif algorithm == 'PPO':
                ppo = PPO()
                ppo.main()
            elif algorithm == 'pbrl':
                delete_episode_data_from_db()
                PBRL.main(iteration)
            elif algorithm == 'reward_model':
                PBRL.learn_reward(iteration)

        elif mode == 'evaluate':
            if algorithm == "dqn":
                PBRL.evaluate(iteration)
            elif algorithm == "pbrl":
                PBRL.evaluate(iteration)

        elif mode == "result":
            if algorithm == 'dqn':
                DQN.get_result(f"{pathConfig.model_save_path}{os.sep}240209_233447{os.sep}24param.pt", ["sks_train_1"])

        elif mode == "query_program":
            app_run()

        elif mode == "label_generator":
            generate_label()

if __name__ == "__main__":
    simulator = Run_Simulator()

    # labeled_data DB 초기화
    # delete_labeled_data_from_db()
    # episode_data DB 초기화
    # delete_episode_data_from_db()

    for i in range(1):
        # mode : query_program, evaluate, learning, result, label_generator
        # algorithm : reward_model, dqn, pbrl

        simulator.main(mode="query_program", algorithm="pbrl", iteration=i)
        simulator.main(mode="label_generator", algorithm="", iteration=i)
        simulator.main(mode="learning", algorithm="reward_model", iteration=i)
        simulator.main(mode="learning", algorithm="pbrl", iteration=i)
        simulator.main(mode="evaluate", algorithm="pbrl", iteration=i)
        # simulator.main(mode="query_program", algorithm="", iteration=i)

