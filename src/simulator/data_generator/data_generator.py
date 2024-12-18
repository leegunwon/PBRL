from src.simulator.Simulator import *
from src.master_db.PBRL_DB_interface import load_episode_data_from_db
from src.master_db.PBRL_DB_interface import save_labeled_data_to_db



def run_simulation_from_trajectory(df):
    lower_bound = random.randint(10, len(df) - Hyperparameters.trajectory_segmentation_size)
    upper_bound = lower_bound + Hyperparameters.trajectory_segmentation_size
    df = df.iloc[:upper_bound].copy()
    df.loc[:, 'trajectory_segmentation'] = False
    df.loc[lower_bound:lower_bound + Hyperparameters.trajectory_segmentation_size - 1, 'trajectory_segmentation'] = True
    trajectory_segmentation = df.iloc[lower_bound:upper_bound, :-1]
    for i in range(len(df)):
        Simulator.step4(df.iloc[i, [-2, -1]])
    setup_time = Simulator.sample_setup_times
    return trajectory_segmentation, setup_time

def generate_label():
    print("labeling start")
    for i in range(Hyperparameters.label_generation_size):
        num = random.sample(range(0, Hyperparameters.episode), 2)

        Simulator.reset(Parameters.datasetId)
        df1 = load_episode_data_from_db(num[0])
        trajectory_segmentation1, setup_time1 = run_simulation_from_trajectory(df1)

        Simulator.reset(Parameters.datasetId)
        df2 = load_episode_data_from_db(num[1])
        trajectory_segmentation2, setup_time2 = run_simulation_from_trajectory(df2)

        if setup_time1 > setup_time2:
            label = 1
        elif setup_time2 > setup_time1:
            label = 0
        else:
            label = 0.5
        temp = [[*trajectory_segmentation1.iloc[i], *trajectory_segmentation2.iloc[i], label] for i in range(Hyperparameters.trajectory_segmentation_size)]
        df = pd.DataFrame(temp)
        save_labeled_data_to_db(df)
    print("labeling end")

