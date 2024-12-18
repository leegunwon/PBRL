import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from src.master_db.PBRL_DB_interface import load_episode_data_from_db
from src.master_db.PBRL_DB_interface import save_labeled_data_to_db
from src.simulator.Simulator import *
import copy

# Dash application 생성
app = dash.Dash(__name__)

prev_clicks_A, prev_clicks_B, prev_clicks_draw, counts = 0.0, 0.0, 0.0, 0
prev_n_clicks_prev_A, prev_n_clicks_next_A, prev_n_clicks_prev_B, prev_n_clicks_next_B = 0, 0, 0, 0
trajectory_segmentation1, trajectory_segmentation2 = None, None
categories = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10"]
size_trajectory_segmentation = Hyperparameters.trajectory_segmentation_size
current_frame_index_A = 4
current_frame_index_B = 4
flow_time1, flow_time2, setup_time1, setup_time2 = 0, 0, 0, 0
fig1, fig2 = None, None

def check_bar(bar, current_frame_index):
    result = False
    for i in range(current_frame_index + 1):
        if f'index_{i}' in bar.legendgroup:
            result = True
    if 'False' in bar.legendgroup:
        result = True
    return result

def cal_performance():
    flow_time, util, makespan, tardiness_time, lateness_time = Simulator.performance_measure_for_pbrl()
    setup_time = Simulator.sample_setup_times
    setup_time_str = seconds_to_days_hours(setup_time)
    flow_time_str = seconds_to_days_hours(flow_time)
    return util, tardiness_time, lateness_time, setup_time_str, flow_time_str

def seconds_to_days_hours(milliseconds):
    seconds = milliseconds / 1000
    days, remainder = divmod(seconds, 86400)  # 1일 = 86400초
    hours = remainder // 3600
    return f"{int(days)}d {int(hours)}h"


def hours_to_days_hours(hours):
    days = hours // 24
    remaining_hours = hours % 24
    return f"{int(days)}d {int(remaining_hours)}h"


def generate_text_content_job_due_date(label, current_frame_index, trajectory_segmentation, indices):
    content = ", ".join([
        f"{label} Job 0{i + 1}: {hours_to_days_hours(trajectory_segmentation.iloc[current_frame_index, idx] * 24) if trajectory_segmentation.iloc[current_frame_index, idx] != 0 else 'DONE'}"
        for i, idx in enumerate(indices)
    ])
    return [html.P(line.strip()) for line in content.split(",")]


def generate_text_content_job_count(label, current_frame_index, trajectory_segmentation, indices):
    content = ", ".join([
        f"{label} Job 0{i + 1}: {trajectory_segmentation.iloc[current_frame_index, idx] if trajectory_segmentation.iloc[current_frame_index, idx] != 0 else 'DONE'}"
        for i, idx in enumerate(indices)
    ])
    return [html.P(line.strip()) for line in content.split(",")]


def generate_gantt_chart(df, label, lower_bound, higher_bound, base_date, max_date):
    Simulator.reset(Parameters.datasetId)
    df = df.iloc[:higher_bound].copy()
    df.loc[:, 'trajectory_segmentation'] = [False] * lower_bound + [f"index_{i}" for i in
                                                                    range(size_trajectory_segmentation)]
    trajectory_segmentation = df.iloc[lower_bound:higher_bound, :-1]
    for i in range(len(df)):
        Simulator.step3(df.iloc[i, [-2, -1]])
    fig = GanttChart.gantt_chart_for_labeling()

    fig.update_layout(
        title={'text': f"{label}", 'font': {'size': 40}},
        yaxis=dict(categoryorder='array', tickfont=dict(size=36), categoryarray=categories),
        xaxis=dict(range=[base_date, max_date], tickfont=dict(size=36)),
        showlegend=False,
    )
    fig.update_xaxes(
        tickformat="%b %d",
    )

    return fig, trajectory_segmentation


app.layout = html.Div(
    style={
        'transform': 'scale(0.66)',
        'transform-origin': 'top left',
        'width': '150%'
    },
    children=[
        html.Div([
            dcc.Graph(id='graph1', style={'width': '100%', 'height': '800px'}),
            html.Div([
                html.Button(id='button_prev_A', n_clicks=0, children='Previous',
                            style={'font-size': '20px', 'width': '200px', 'height': '50px', 'margin': '20px'}),
                html.Button(id='button_next_A', n_clicks=0, children='Next',
                            style={'font-size': '20px', 'width': '200px', 'height': '50px', 'margin': '20px'})
            ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '10px'}),
            dcc.Graph(id='graph2', style={'width': '100%', 'height': '800px'}),
            html.Div([
                html.Button(id='button_prev_B', n_clicks=0, children='Previous',
                            style={'font-size': '20px', 'width': '200px', 'height': '50px', 'margin': '20px'}),
                html.Button(id='button_next_B', n_clicks=0, children='Next',
                            style={'font-size': '20px', 'width': '200px', 'height': '50px', 'margin': '20px'})
            ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '10px'}),
            html.Div([
                html.Button(id='button_win_A', n_clicks=0, children='A is better',
                            style={'font-size': '28px', 'width': '300px', 'height': '70px', 'margin': '20px'}),
                html.Button(id='button_draw', n_clicks=0, children='draw',
                            style={'font-size': '28px', 'width': '300px', 'height': '70px', 'margin': '20px'}),
                html.Button(id='button_win_B', n_clicks=0, children='B is better',
                            style={'font-size': '28px', 'width': '300px', 'height': '70px', 'margin': '20px'}),
            ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'margin-top': '20px'})
        ], style={'width': '65%', 'display': 'inline-block', 'vertical-align': 'top'}),

        html.Div([
            html.H4("Impact of Actions in Chart A"),
            html.Div(id='text1'),
            html.H4("Remaining Jobs Due Date (A)"),
            html.Div(id='text3'),
            html.H4("Remaining Jobs Count (A)"),
            html.Div(id='text4'),
        ], style={
            'position': 'absolute', 'top': '50px', 'left': '1950px', 'width': '15%', 'padding': '5px',
            'border': '2px solid #ddd', 'background-color': 'rgba(255, 255, 255, 0.8)', 'z-index': '10'
        }),

        html.Div([
            html.H4("Impact of Actions in Chart B"),
            html.Div(id='text2'),
            html.H4("Remaining Jobs Due Date (B)"),
            html.Div(id='text5'),
            html.H4("Remaining Jobs Count (B)"),
            html.Div(id='text6'),
        ], style={
            'position': 'absolute', 'top': '950px', 'left': '1950px', 'width': '15%', 'padding': '5px',
            'border': '2px solid #ddd', 'background-color': 'rgba(255, 255, 255, 0.8)', 'z-index': '10'
        }),
    ]
)


@app.callback(
    [Output('graph1', 'figure'),
     Output('graph2', 'figure'),
     Output('text1', 'children'),
     Output('text2', 'children'),
     Output('text3', 'children'),
     Output('text4', 'children'),
     Output('text5', 'children'),
     Output('text6', 'children')],
    [Input('button_win_A', 'n_clicks'),
     Input('button_win_B', 'n_clicks'),
     Input('button_draw', 'n_clicks'),
     Input('button_prev_A', 'n_clicks'),
     Input('button_next_A', 'n_clicks'),
     Input('button_prev_B', 'n_clicks'),
     Input('button_next_B', 'n_clicks')]
)
def update_graph(n_clicks_A, n_clicks_B, n_clicks_draw, n_clicks_prev_A, n_clicks_next_A, n_clicks_prev_B,
                 n_clicks_next_B):
    global prev_clicks_A, prev_clicks_B, prev_clicks_draw, current_frame_index_A, current_frame_index_B
    global prev_n_clicks_prev_A, prev_n_clicks_next_A, prev_n_clicks_prev_B, prev_n_clicks_next_B
    global categories, size_trajectory_segmentation, trajectory_segmentation1, trajectory_segmentation2
    global flow_time1, flow_time2, setup_time1, setup_time2, fig1, fig2

    # 의사결정 계산 코드
    prev_clicks_draw = n_clicks_draw - prev_clicks_draw
    label_a, label_b = n_clicks_A - prev_clicks_A + prev_clicks_draw / 2, n_clicks_B - prev_clicks_B + prev_clicks_draw / 2
    prev_clicks_A, prev_clicks_B, prev_clicks_draw = n_clicks_A, n_clicks_B, n_clicks_draw
    click_p_a, click_n_a = n_clicks_prev_A - prev_n_clicks_prev_A, n_clicks_next_A - prev_n_clicks_next_A
    click_p_b, click_n_b = n_clicks_prev_B - prev_n_clicks_prev_B, n_clicks_next_B - prev_n_clicks_next_B
    prev_n_clicks_prev_A, prev_n_clicks_next_A, prev_n_clicks_prev_B, prev_n_clicks_next_B = n_clicks_prev_A, n_clicks_next_A, n_clicks_prev_B, n_clicks_next_B

    # current_frame_index 업데이트
    current_frame_index_A = max(0,
                                min(current_frame_index_A + (click_n_a - click_p_a), size_trajectory_segmentation - 1))
    current_frame_index_B = max(0,
                                min(current_frame_index_B + (click_n_b - click_p_b), size_trajectory_segmentation - 1))

    if label_a + label_b == 1:
        temp = [[*trajectory_segmentation1.iloc[i], *trajectory_segmentation2.iloc[i], label_b] for i in
                range(size_trajectory_segmentation)]
        df = pd.DataFrame(temp)
        save_labeled_data_to_db(df)
        current_frame_index_A = 4
        current_frame_index_B = 4

    if click_p_a + click_n_a + click_p_b + click_n_b == 0:
        Simulator.reset(Parameters.datasetId)
        num = random.sample(range(0, Hyperparameters.episode), 2)
        df1 = load_episode_data_from_db(num[0])
        lower_bound = random.randint(10, len(df1) - size_trajectory_segmentation - 1)
        higher_bound = lower_bound + size_trajectory_segmentation

        base_date = GanttChart.gantt_history['Start'].min()
        max_date = base_date + pd.Timedelta(hours=Parameters.plan_horizon + 60)

        fig1, trajectory_segmentation1 = generate_gantt_chart(df1, "chart A", lower_bound, higher_bound, base_date, max_date)

        util1, tardiness_time1, lateness_time1, setup_time1, flow_time1 = cal_performance()

        df2 = load_episode_data_from_db(num[1])
        lower_bound = lower_bound + random.randint(-5, 5)
        lower_bound = min(max(lower_bound, 10), len(df2) - size_trajectory_segmentation - 1)
        higher_bound = lower_bound + size_trajectory_segmentation

        fig2, trajectory_segmentation2 = generate_gantt_chart(df2, "chart B", lower_bound, higher_bound, base_date, max_date)

        util2, tardiness_time2, lateness_time2, setup_time2, flow_time2 = cal_performance()


        text1_content = f"Lateness: {lateness_time1}, Tardiness: {tardiness_time1}, flow_time: {flow_time1}, Setup Time: {setup_time1}"
        text1_content_lines = [html.P(line.strip()) for line in text1_content.split(",")]
        text2_content = f"Lateness: {lateness_time2}, Tardiness: {tardiness_time2}, flow_time: {flow_time2}, Setup Time: {setup_time2}"
        text2_content_lines = [html.P(line.strip()) for line in text2_content.split(",")]
        text3_content_lines = generate_text_content_job_due_date("A", current_frame_index_A, trajectory_segmentation1,
                                                                 range(40, 45))
        text4_content_lines = generate_text_content_job_count("A", current_frame_index_A, trajectory_segmentation1,
                                                              range(45, 50))
        text5_content_lines = generate_text_content_job_due_date("B", current_frame_index_B, trajectory_segmentation2,
                                                                 range(40, 45))
        text6_content_lines = generate_text_content_job_count("B", current_frame_index_B, trajectory_segmentation2,
                                                              range(45, 50))

        return fig1, fig2, text1_content_lines, text2_content_lines, text3_content_lines, text4_content_lines, text5_content_lines, text6_content_lines

    elif click_p_a + click_n_a == 1:
        temp_fig1 = copy.copy(fig1)
        temp_fig1.data = [bar for bar in temp_fig1.data if check_bar(bar, current_frame_index_A)]

        text3_content_lines = generate_text_content_job_due_date("A", current_frame_index_A, trajectory_segmentation1,
                                                                 range(40, 45))
        text4_content_lines = generate_text_content_job_count("A", current_frame_index_A, trajectory_segmentation1,
                                                              range(45, 50))

        return temp_fig1, dash.no_update, dash.no_update, dash.no_update, text3_content_lines, text4_content_lines, dash.no_update, dash.no_update

    elif click_p_b + click_n_b == 1:
        temp_fig2 = copy.copy(fig2)
        temp_fig2.data = [bar for bar in temp_fig2.data if check_bar(bar, current_frame_index_B)]

        text5_content_lines = generate_text_content_job_due_date("B", current_frame_index_B, trajectory_segmentation2,
                                                                 range(40, 45))
        text6_content_lines = generate_text_content_job_count("B", current_frame_index_B, trajectory_segmentation2,
                                                              range(45, 50))

        return dash.no_update, temp_fig2, dash.no_update, dash.no_update, dash.no_update, dash.no_update, text5_content_lines, text6_content_lines


def app_run():
    app.run_server(debug=True)
