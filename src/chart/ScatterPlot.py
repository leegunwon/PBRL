import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from statsmodels.api import OLS, add_constant
from src.common.pathConfig import pathConfig
from src.chart.ChartParameters import ChartParameters
import os

def calculate_r_squared(x, y):
    x = pd.to_numeric(x, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Numpy 배열로 변환 (statsmodels 요구사항)
    x = add_constant(x.to_numpy())
    y = y.to_numpy()

    # 회귀 모델 적합
    model = OLS(y, x).fit()
    return model.rsquared

def cumulative_reward_scatter_plot_plotly(df, count):
    fig = px.scatter(
        df, x='Episodes', y='Cumulative Reward', trendline='ols',
        title='Change in cumulative reward over episodes',
        labels={"x": "Episodes", "Cumulative Reward": "Cumulative Reward"},
        opacity=0.7
    )
    fig.update_traces(marker=dict(color='blue'))  # 점 색상 고정
    fig.update_traces(line=dict(color='red', width=2), selector=dict(mode='lines'))  # 추세선 강조
    fig.update_layout(title=dict(
        text="Change in cumulative reward over episodes",  # 제목 텍스트
        font=dict(size=30)),
        xaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=24)),  # x축 글꼴 크기
        yaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=24)))

    r_squared = calculate_r_squared(df['Episodes'], df['Cumulative Reward'])

    fig.write_html(f"{pathConfig.pbrl_result_plotly_path}{os.sep}{count}Change_in_cumulative_reward_{r_squared:.2f}.html")

    return r_squared

def cumulative_reward_scatter_plot_seaborn(df, count, r_squared):
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Episodes', y='Cumulative Reward', alpha=0.5, color="blue")
    sns.regplot(data=df, x='Episodes', y='Cumulative Reward', scatter=False, color='red')
    plt.title("Change in cumulative reward over episodes", fontsize=24)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Cumulative Reward", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{pathConfig.pbrl_result_seaborn_path}{os.sep}{count}seaborn_cumulative_reward_{r_squared:.2f}.png")

def cumulative_reward_scatter_plot_matplotlib(df, count, r_squared):
    plt.figure(figsize=(15, 10))
    plt.scatter(df['Episodes'], df['Cumulative Reward'], alpha=0.5, color='blue')
    plt.title("Change in cumulative reward over episodes", fontsize=20)
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Cumulative Reward", fontsize=20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(
        f"{pathConfig.pbrl_result_matplotlib_path}{os.sep}{count}matplotlib_cumulative_reward_{r_squared:.2f}.png")

def correlation_between_util_score_scatter_plot_plotly(df, count):
    fig = px.scatter(
        df, x='Cumulative Reward', y='Utilization', trendline='ols',
        title='Correlation between cumulative reward and util',
        labels={"Cumulative Reward": "Cumulative Reward", "Utilization": "Utilization"},
        opacity=0.7
    )
    fig.update_traces(marker=dict(color='green'))
    fig.update_traces(line=dict(color='red', width=2), selector=dict(mode='lines'))
    fig.update_layout(title=dict(
        text="Correlation between cumulative reward and utilization",
        font=dict(size=30)),
        xaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=24)),
        yaxis=dict(title=dict(font=dict(size=28)), tickfont=dict(size=24)))

    r_squared = calculate_r_squared(df['Cumulative Reward'], df['Utilization'])
    fig.write_html(
        f"{pathConfig.pbrl_result_plotly_path}{os.sep}{count}Correlation_between_reward_and_utilization_{r_squared:.2f}.html")
    
    return r_squared

def correlation_between_util_score_scatter_plot_seaborn(df, count, r_squared):
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Cumulative Reward', y='Utilization', alpha=0.5, color="green")
    sns.regplot(data=df, x='Cumulative Reward', y='Utilization', scatter=False, color='red')
    plt.title("Correlation between cumulative reward and util", fontsize=24)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.xlabel("Cumulative Reward", fontsize=16)
    plt.ylabel("Utilization", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{pathConfig.pbrl_result_seaborn_path}{os.sep}{count}seaborn_correlation_{r_squared:.2f}.png")

def correlation_between_util_score_scatter_plot_matplotlib(df, count, r_squared):
    plt.figure(figsize=(15, 10))
    plt.scatter(df['Cumulative Reward'], df['Utilization'], alpha=0.5, color='green')
    plt.title("Correlation between cumulative reward and util", fontsize=20)
    plt.xlabel("Cumulative Reward", fontsize=20)
    plt.ylabel("Utilization", fontsize=20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(
        f"{pathConfig.pbrl_result_matplotlib_path}{os.sep}{count}matplotlib_correlation_{r_squared:.2f}.png")

def utilization_scatter_plot_plotly(df, count):
    fig = px.scatter(
        df, x='Episodes', y='Utilization', trendline='ols',
        title='Change in utilization over episodes',
        labels={"x": "Episodes", "Utilization": "Utilization"},
        opacity=0.7
    )
    fig.update_traces(marker=dict(color='black'))
    fig.update_traces(line=dict(color='red', width=2), selector=dict(mode='lines'))
    r_squared = calculate_r_squared(df['Episodes'], df['Utilization'])
    fig.write_html(
        f"{pathConfig.pbrl_result_plotly_path}{os.sep}{count}Change_in_utilization_{r_squared:.2f}.html")
    
    return r_squared

def utilization_scatter_plot_seaborn(df, count, r_squared):
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Episodes', y='Utilization', alpha=0.5, color="black")
    sns.regplot(data=df, x='Episodes', y='Utilization', scatter=False, color='red')
    plt.title("Change in utilization over episodes", fontsize=24)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Utilization", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{pathConfig.pbrl_result_seaborn_path}{os.sep}{count}seaborn_utilization_change_{r_squared:.2f}.png")

def utilization_scatter_plot_matplotlib(df, count, r_squared):
    plt.figure(figsize=(15, 10))
    plt.scatter(df['Episodes'], df['Utilization'], alpha=0.5, color='black')
    plt.title("Change in utilization over episodes", fontsize=20)
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Utilization", fontsize=20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(
        f"{pathConfig.pbrl_result_matplotlib_path}{os.sep}{count}matplotlib_utilization_{r_squared:.2f}.png")

def loss_scatter_plot_plotly(df, count):
    fig = px.scatter(
        df, x='Episodes', y='Loss', trendline='ols',
        title='Change in loss over episodes',
        labels={"x": "Episodes", "loss": "Loss"},
        opacity=0.7
    )
    fig.update_traces(marker=dict(color='purple'))
    fig.update_traces(line=dict(color='red', width=2), selector=dict(mode='lines'))
    r_squared = calculate_r_squared(df['Episodes'], df['Loss'])
    fig.write_html(
        f"{pathConfig.pbrl_result_plotly_path}{os.sep}{count}Change_in_loss_{r_squared:.2f}.html")
    
    return r_squared

def loss_scatter_plot_seaborn(df, count, r_squared):
    sns.set(style="whitegrid", palette="pastel")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Episodes', y='Loss', alpha=0.5, color="purple")
    sns.regplot(data=df, x='Episodes', y='Loss', scatter=False, color='red')
    plt.title("Change in loss over episodes", fontsize=24)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.xlabel("Episodes", fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        f"{pathConfig.pbrl_result_seaborn_path}{os.sep}{count}seaborn_loss_change_{r_squared:.2f}.png")

def loss_scatter_plot_matplotlib(df, count, r_squared):
    plt.figure(figsize=(15, 10))
    plt.scatter(df['Episodes'], df['Loss'], alpha=0.5, color='purple')
    plt.title("Change in loss over episodes", fontsize=20)
    plt.xlabel("Episodes", fontsize=20)
    plt.ylabel("Loss", fontsize=20)
    plt.tick_params(axis='x', labelsize=14)
    plt.tick_params(axis='y', labelsize=14)
    plt.savefig(
        f"{pathConfig.pbrl_result_matplotlib_path}{os.sep}{count}matplotlib_loss_{r_squared:.2f}.png")

def pbrl_Learning_scatter_plot(score_list, util_list, loss_list, count):
    r_squared = 0
    if ChartParameters.cumulative_reward_plot:
        df = pd.DataFrame({'Episodes': list(range(1, len(score_list) + 1)), 'Cumulative Reward': score_list})
        if ChartParameters.plotly_chart:
            r_squared = cumulative_reward_scatter_plot_plotly(df, count)
        if ChartParameters.seaborn_chart:
            cumulative_reward_scatter_plot_seaborn(df, count, r_squared)
        if ChartParameters.matplotlib_chart:
            cumulative_reward_scatter_plot_matplotlib(df, count, r_squared)

    if ChartParameters.correlation_between_util_score_plot:
        df = pd.DataFrame({'Cumulative Reward': score_list, 'Utilization': util_list})
        if ChartParameters.plotly_chart:
            r_squared = correlation_between_util_score_scatter_plot_plotly(df, count)
        if ChartParameters.seaborn_chart:
            correlation_between_util_score_scatter_plot_seaborn(df, count, r_squared)
        if ChartParameters.matplotlib_chart:
            correlation_between_util_score_scatter_plot_matplotlib(df, count, r_squared)

    if ChartParameters.utilization_plot:
        df = pd.DataFrame({'Episodes': list(range(1, len(util_list) + 1)), 'Utilization': util_list})
        if ChartParameters.plotly_chart:
            r_squared = utilization_scatter_plot_plotly(df, count)
        if ChartParameters.seaborn_chart:
            utilization_scatter_plot_seaborn(df, count, r_squared)
        if ChartParameters.matplotlib_chart:
            utilization_scatter_plot_matplotlib(df, count, r_squared)

    if ChartParameters.loss_plot:
        df = pd.DataFrame({'Episodes': list(range(1, len(loss_list) + 1)), 'Loss': loss_list})
        if ChartParameters.plotly_chart:
            r_squared = loss_scatter_plot_plotly(df, count)
        if ChartParameters.seaborn_chart:
            loss_scatter_plot_seaborn(df, count, r_squared)
        if ChartParameters.matplotlib_chart:
            loss_scatter_plot_matplotlib(df, count, r_squared)
