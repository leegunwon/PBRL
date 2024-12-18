import os


class pathConfig:
    absolute_path = ""
    log_path = ""
    model_save_path = ""
    pickle_data_path = ""
    pickle_simulator_data_path = ""
    gantt_save_path = ""
    simulator_result_gantt_chart_path = ""
    reward_learning_data_path = ""
    reward_model_params_path = ""
    reward_model_learning_result_path = ""
    reinforcement_model_params_path = ""
    pbrl_result_plotly_path = ""
    pbrl_result_seaborn_path = ""
    pbrl_result_matplotlib_path = ""

    @classmethod
    def set_absolute_path(cls):
        script_path = os.path.abspath(__file__)
        dir_path = os.path.dirname(script_path)
        cls.absolute_path = os.path.dirname(dir_path)  # src 파일 경로
        cls.log_path = f"{cls.absolute_path}{os.sep}log_data"
        cls.model_save_path = f"{cls.absolute_path}{os.sep}params_data"
        cls.pickle_data_path = f"{cls.absolute_path}{os.sep}master_db{os.sep}pickleDBData"
        cls.simulator_result_gantt_chart_path = f"{cls.absolute_path}{os.sep}simulator_result{os.sep}gantt_chart"
        cls.pickle_simulator_data_path = f"{cls.absolute_path}{os.sep}master_db{os.sep}pickleSimulatorData"
        cls.reward_learning_data_path = f"{cls.absolute_path}{os.sep}save_data{os.sep}reward_learning_data"
        cls.reward_model_params_path = f"{cls.absolute_path}{os.sep}params_data{os.sep}reward_model_params"
        cls.reinforcement_model_params_path = f"{cls.absolute_path}{os.sep}params_data{os.sep}reinforcement_learning"
        cls.pbrl_result_plotly_path = f"{cls.absolute_path}{os.sep}simulator_result{os.sep}pbrl_learning_result{os.sep}plotly"
        cls.pbrl_result_seaborn_path = f"{cls.absolute_path}{os.sep}simulator_result{os.sep}pbrl_learning_result{os.sep}seaborn"
        cls.pbrl_result_matplotlib_path = f"{cls.absolute_path}{os.sep}simulator_result{os.sep}pbrl_learning_result{os.sep}matplotlib"
        cls.reward_model_learning_result_path = f"{cls.absolute_path}{os.sep}simulator_result{os.sep}reward_model_learning_result"

        try:
            for i in range(-1, 10):
                os.makedirs(f"{cls.reinforcement_model_params_path}{os.sep}{i}", exist_ok=True)
            os.makedirs(cls.simulator_result_gantt_chart_path, exist_ok=True)
            os.makedirs(cls.pbrl_result_plotly_path, exist_ok=True)
            os.makedirs(cls.pbrl_result_seaborn_path, exist_ok=True)
            os.makedirs(cls.pbrl_result_matplotlib_path, exist_ok=True)
            os.makedirs(cls.reward_model_learning_result_path, exist_ok=True)
        except Exception as e:
            print(f"디렉토리 생성 중 오류 발생: {e}")