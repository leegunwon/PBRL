import os


class pathConfig:
    absolute_path = ""
    log_path = ""
    model_save_path = ""
    pickle_data_path = ""
    pickle_simulator_data_path = ""
    gantt_save_path = ""
    simulator_result_path = ""
    reward_learning_data_path = ""
    reward_model_params_path = ""
    labeled_data_path = ""
    unlabeled_data_path = ""
    reinforcement_model_params_path = ""
    PBRL_result_chart_path = ""
    PBRL_result_gantt_chart_path = ""
    pathh =""
    os = "/"

    @classmethod
    def set_absolute_path(cls):
        script_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(script_path)
        cls.absolute_path = os.path.dirname(dir_path)  # src 파일 경로
        cls.log_path = f"{cls.absolute_path}{os.sep}log_data"
        cls.model_save_path = f"{cls.absolute_path}{os.sep}params_data"
        cls.pickle_data_path = f"{cls.absolute_path}{os.sep}master_db{os.sep}pickleDBData"
        # cls.gantt_save_path = f"{cls.absoulte_path}/"
        cls.simulator_result_path = f"{cls.absolute_path}{os.sep}simulator_result"
        cls.pickle_simulator_data_path = f"{cls.absolute_path}{os.sep}master_db{os.sep}pickleSimulatorData"
        cls.reward_learning_data_path = f"{cls.absolute_path}{os.sep}save_data{os.sep}reward_learning_data"
        cls.reward_model_params_path = f"{cls.absolute_path}{os.sep}params_data{os.sep}reward_model_params"
        cls.reinforcement_model_params_path = f"{cls.absolute_path}{os.sep}params_data{os.sep}reinforcement_learning"
        cls.labeled_data_path = f"{cls.absolute_path}{os.sep}save_data{os.sep}labeled_data"
        cls.unlabeled_data_path = f"{cls.absolute_path}{os.sep}save_data{os.sep}unlabeled_data"
        cls.PBRL_result_chart_path = f"{cls.absolute_path}{os.sep}simulator_result{os.sep}PBRL_result"
        cls.PBRL_result_gantt_chart_path = f"{cls.absolute_path}{os.sep}simulator_result{os.sep}PBRL_result{os.sep}gantt_result"
        cls.pathh = f"{cls.absolute_path}"