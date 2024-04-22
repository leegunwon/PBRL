import os
import flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from flask import request
import random
from src.common.Parameters import *
from src.simulator.GanttChart import *
from src.simulator.Simulator import *
from src.common.pathConfig import *
# to_do list
# 데이터를 받아오는 방법
#
#

# Dash 애플리케이션 생성
app = dash.Dash(__name__)

prev_clicks_A, prev_clicks_B, prev_clicks_draw, counts = 0.0, 0.0, 0.0, 0
categories = ["M10", "M9", "M8", "M7", "M6", "M5", "M4", "M3", "M2", "M1"]
size_sample_action = Hyperparameters.size_sample_action
# 레이아웃 설정
app.layout = html.Div([
    dcc.Graph(id='graph1', style={'display': 'inline-block'}),
    dcc.Graph(id='graph2', style={'display': 'inline-block'}),
    html.Div([
        html.Div([
            html.Button(id='button_win_A', n_clicks=0, children='Win_A',
                        style={'font-size': '20px', 'width': '200px', 'margin': 'auto', 'display': 'block'}),
        ], style={'display': 'inline-block', 'margin-right': '10px'}),
        html.Div([
            html.Button(id='button_draw', n_clicks=0, children='draw',
                        style={'font-size': '20px', 'width': '200px', 'margin': 'auto', 'display': 'block'}),
        ], style={'display': 'inline-block', 'margin-right': '10px'}),
        html.Div([
            html.Button(id='button_win_B', n_clicks=0, children='Win_B',
                        style={'font-size': '20px', 'width': '200px', 'margin': 'auto', 'display': 'block'}),
        ], style={'display': 'inline-block', 'margin-right': '10px'}),
    ], style={'text-align': 'center'})
])


# 그래프 콜백 함수 정의
@app.callback(
    [Output('graph1', 'figure'),
     Output('graph2', 'figure')],
    [Input('button_win_A', 'n_clicks'),
     Input('button_win_B', 'n_clicks'),
     Input('button_draw', 'n_clicks'), ]
)

def update_graph(n_clicks_A, n_clicks_B, n_clicks_draw):
    global prev_clicks_A, prev_clicks_B, prev_clicks_draw, categories, size_sample_action

    prev_clicks_draw = n_clicks_draw - prev_clicks_draw
    label_A, label_B = n_clicks_A - prev_clicks_A + prev_clicks_draw / 2, n_clicks_B - prev_clicks_B + prev_clicks_draw / 2
    prev_clicks_A, prev_clicks_B, prev_clicks_draw = n_clicks_A, n_clicks_B, n_clicks_draw

    # 파라미터화 해야함
    Simulator.reset(Parameters.datasetId)
    num = random.sample(range(0, Hyperparameters.episode), 2)
    df1 = pd.read_csv(f"{pathConfig.unlabeled_data_path}{os.sep}inputs{num[0]}.csv", index_col=0)
    lower_bound = random.randint(10, len(df1) - size_sample_action)
    higher_bound = lower_bound + size_sample_action
    df1 = df1.iloc[:higher_bound]
    df1['sample_action'] = [False] * lower_bound + [True] * size_sample_action
    sample_action1 = df1.iloc[lower_bound:higher_bound, [0, 1, 2, 3, 4]]
    for i in range(len(df1)):
        Simulator.step3(df1.iloc[i, [4, 5]])
    fig1 = GanttChart.csv_to_gantt_chart() # action 부분만 indexing 함

    base_date = GanttChart.gantt_history['Start'].min()
    max_date = base_date + pd.Timedelta(hours=Parameters.plan_horizon + 40)

    Flow_time, machine_util, util, makespan, tardiness, lateness, t_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time, rtf = Simulator.performance_measure()

    fig1.update_layout(title=f"chart_A flow time {Flow_time}   machine_util {machine_util:.2f}   util {util:.2f}   makespan {makespan}   tardiness {tardiness}   lateness {lateness}   t_max {t_max}   traj_len {higher_bound}", yaxis=dict(categoryorder='array',
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
    sample_action2 = df2.iloc[lower_bound:higher_bound, [0, 1, 2, 3, 4]]
    for i in range(len(df2)):
        Simulator.step3(df2.iloc[i, [4, 5]])
    fig2 = GanttChart.csv_to_gantt_chart() # action 부분만 indexing 함

    Flow_time, machine_util, util, makespan, tardiness, lateness, t_max, q_time_true, q_time_false, q_job_t, q_job_f, q_time, rtf = Simulator.performance_measure()

    fig2.update_layout(title=f"chart_B flow time {Flow_time}   machine_util {machine_util:.2f}   util {util:.2f}   makespan {makespan}   tardiness {tardiness}   lateness {lateness}   t_max {t_max}   traj_len {higher_bound}", yaxis=dict(categoryorder='array',
            categoryarray=categories), xaxis=dict(range=[base_date, max_date]))

    if (label_A + label_B == 1):
        file_path = f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
        else:
            df = pd.DataFrame([], columns=[str(k) for k in range(len(df1) + 2)])
        for i in range(size_sample_action):
            new_df = pd.DataFrame([*sample_action1.iloc[i], *sample_action2.iloc[i], label_A, label_B]).T
            new_df.columns = [str(j) for j in range(12)]
            df = pd.concat([df, new_df], axis = 0)
        df.to_csv(f"{pathConfig.labeled_data_path}{os.sep}labeled_data.csv", index=True)


    return fig1, fig2


def app_run():
    app.run_server(debug=True)