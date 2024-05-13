from src.common.Parameters import *
from src.simulator.GanttChart import *
from src.simulator.Simulator import *
from src.common.pathConfig import *
from src.common.pathConfig import *
def generate_label():
    print("labeling on")
    size_sample_action = Hyperparameters.size_sample_action
    file_path = f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
    else:
        df = pd.DataFrame([], columns=[str(k) for k in range((Hyperparameters.ds + Hyperparameters.da)*2 + 1)])

    for i in range(Hyperparameters.number_of_labeling):
        # 파라미터화 해야함
        Simulator.reset(Parameters.datasetId)
        num = random.sample(range(0, Hyperparameters.episode), 2)
        df1 = pd.read_csv(f"{pathConfig.unlabeled_data_path}{os.sep}inputs{num[0]}.csv", index_col=0)
        lower_bound = random.randint(10, len(df1) - size_sample_action)
        upper_bound = lower_bound + size_sample_action
        df1 = df1.iloc[:upper_bound]
        df1['sample_action'] = [False] * lower_bound + [True] * size_sample_action
        sample_action1 = df1.iloc[lower_bound:upper_bound, range(len(df1.T)-1)]
        for i in range(len(df1)):
            Simulator.step4(df1.iloc[i, [-2, -1]])
        setup_time1 = Simulator.sample_setup_times

        Simulator.reset(Parameters.datasetId)
        # 데이터 불러서 sample labeling 작업 수행함
        df2 = pd.read_csv(f"{pathConfig.unlabeled_data_path}{os.sep}inputs{num[1]}.csv", index_col=0)
        lower_bound = lower_bound + random.randint(-5, 5)
        lower_bound = lower_bound if lower_bound <= len(df2) - size_sample_action -1 else len(df2) - size_sample_action -1
        lower_bound = lower_bound if lower_bound >= 10 else 10
        upper_bound = lower_bound + size_sample_action
        df2 = df2.iloc[:upper_bound]
        df2['sample_action'] = [False] * lower_bound + [True] * size_sample_action
        sample_action2 = df2.iloc[lower_bound:upper_bound, range(len(df2.T)-1)]
        for i in range(len(df2)):
            Simulator.step4(df2.iloc[i, [-2, -1]])
        setup_time2 = Simulator.sample_setup_times

        if setup_time1 > setup_time2:
            label = 1
        elif setup_time2 > setup_time1:
            label = 0
        else:
            label = 0.5

        for i in range(size_sample_action):
            new_df = pd.DataFrame([*sample_action1.iloc[i], *sample_action2.iloc[i], label]).T
            new_df.columns = [str(j) for j in range(len(new_df.columns))]
            df = pd.concat([df, new_df], axis=0)

    df.to_csv(f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv", index=True)
    print("labeling end")



def delete_file(file_path):
    # 파일이 존재하는지 확인
    if os.path.exists(file_path):
        # 파일 삭제
        os.remove(file_path)
        print(f"{file_path} 파일이 삭제되었습니다.")
    else:
        # 파일이 없을 경우
        print(f"{file_path} 파일을 찾을 수 없습니다.")

def file_reset():
    delete_file(pathConfig.reward_model_params_path)
    delete_file(pathConfig.reinforcement_model_params_path)
    delete_file(pathConfig.labeled_data_path)
