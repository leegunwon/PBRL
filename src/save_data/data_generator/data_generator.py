from src.common.Parameters import *
from src.simulator.GanttChart import *
from src.simulator.Simulator import *
from src.common.pathConfig import *

def generate_label():
    categories = ["M10", "M9", "M8", "M7", "M6", "M5", "M4", "M3", "M2", "M1"]
    size_sample_action = Hyperparameters.size_sample_action

    # 파라미터화 해야함
    Simulator.reset(Parameters.datasetId)
    num = random.sample(range(0, Hyperparameters.episode), 2)
    df1 = pd.read_csv(f"{pathConfig.unlabeled_data_path}{os.sep}inputs{num[0]}.csv", index_col=0)
    lower_bound = random.randint(10, len(df1) - size_sample_action)
    higher_bound = lower_bound + size_sample_action
    df1 = df1.iloc[:higher_bound]
    df1['sample_action'] = [False] * lower_bound + [True] * size_sample_action
    sample_action1 = df1.iloc[lower_bound:higher_bound, range(23)]
    for i in range(len(df1)):
        Simulator.step3(df1.iloc[i, [-2, -1]])
    fig1 = GanttChart.csv_to_gantt_chart() # action 부분만 indexing 함

    base_date = GanttChart.gantt_history['Start'].min()
    max_date = base_date + pd.Timedelta(hours=Parameters.plan_horizon + 40)

    setup_time = [bar.x[0] for bar in fig1.data if ('setup' in bar.legendgroup and bar.marker.pattern.shape == "/")]
    setup_time1 = sum(setup_time)

    fig1.update_layout(title=f"chart_A setup_time : {setup_time1}", yaxis=dict(categoryorder='array',
                        categoryarray=categories), xaxis=dict(range=[base_date, max_date]))
    Simulator.reset(Parameters.datasetId)

    # 데이터 불러서 sample labeling 작업 수행함
    df2 = pd.read_csv(f"{pathConfig.unlabeled_data_path}{os.sep}inputs{num[1]}.csv", index_col=0)
    lower_bound = lower_bound + random.randint(-5, 5)
    lower_bound = lower_bound if lower_bound <= len(df2) - size_sample_action -1 else len(df2) - size_sample_action -1
    lower_bound = lower_bound if lower_bound >= 10 else 10
    higher_bound = lower_bound + size_sample_action
    df2 = df2.iloc[:higher_bound]
    df2['sample_action'] = [False] * lower_bound + [True] * size_sample_action
    sample_action2 = df2.iloc[lower_bound:higher_bound, range(23)]
    for i in range(len(df2)):
        Simulator.step3(df2.iloc[i, [-2, -1]])
    fig2 = GanttChart.csv_to_gantt_chart() # action 부분만 indexing 함
    setup_time = [bar.x[0] for bar in fig2.data if ('setup' in bar.legendgroup and bar.marker.pattern.shape == "/")]
    setup_time2 = sum(setup_time)
    Flow_time, machine_util, util, makespan, tardiness, lateness, t_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time, rtf = Simulator.performance_measure()

    fig2.update_layout(title=f"chart_B setup_time : {setup_time2}", yaxis=dict(categoryorder='array',
            categoryarray=categories), xaxis=dict(range=[base_date, max_date]))

    if setup_time1 > setup_time2:
        label = 0
    elif setup_time2 > setup_time1:
        label = 1
    else:
        label = 0.5

    file_path = f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
    else:
        df = pd.DataFrame([], columns=[str(k) for k in range(sample_action1.shape[1]+sample_action2.shape[1] + 1)])
    for i in range(size_sample_action):
        new_df = pd.DataFrame([*sample_action1.iloc[i], *sample_action2.iloc[i], label]).T
        new_df.columns = [str(j) for j in range(len(new_df.columns))]
        df = pd.concat([df, new_df], axis = 0)
    df.to_csv(f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv", index=True)
