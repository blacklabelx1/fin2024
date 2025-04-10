import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt

# 데이터 준비
upjong_price = pd.read_csv('https://raw.githubusercontent.com/pinkocto/DataHub/main/data/upjong_price_220113_240524.csv')  # 업종 데이터
upjong_price['날짜'] = pd.to_datetime(upjong_price['날짜'])
upjong_price = upjong_price.set_index('날짜')
daily_simple_returns = upjong_price.pct_change(1).dropna()  # 일별 수익률 계산

weekly_returns = daily_simple_returns.resample("W").sum()
monthly_returns = daily_simple_returns.resample("M").sum()
quarterly_returns = daily_simple_returns.resample('Q').sum()

stock_upjong_info = pd.read_csv('https://raw.githubusercontent.com/pinkocto/DataHub/main/data/stock_upjong_ticker_info.csv', dtype={'종목코드': str})

returns = upjong_price.pct_change().dropna()
returns_T = returns.T
similarity_matrix = cosine_similarity(returns_T)

tsne = TSNE(n_components=2, perplexity=23.0, learning_rate=10, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(similarity_matrix)

# 종목 데이터 준비
df_stock2 = pd.read_csv('https://raw.githubusercontent.com/pinkocto/DataHub/main/data/stock_price_220101_240524.csv').set_index('날짜')

returns_stock = df_stock2.pct_change().dropna()
returns_stock_T = returns_stock.T
similarity_matrix2 = cosine_similarity(returns_stock_T)
similarity_matrix2 += 1e-9

def tsne_kl_divergence(similarity_matrix, perplexity, learning_rate):
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, n_iter=1000, random_state=42)
    tsne_results = tsne.fit_transform(similarity_matrix)

    def calculate_probability_distribution(matrix):
        row_sums = matrix.sum(axis=1, keepdims=True)
        return matrix / row_sums

    original_prob_dist = calculate_probability_distribution(similarity_matrix)
    tsne_prob_dist = calculate_probability_distribution(cosine_similarity(tsne_results))

    original_prob_dist = original_prob_dist.flatten()
    tsne_prob_dist = tsne_prob_dist.flatten()

    original_prob_dist = np.clip(original_prob_dist, 1e-9, None)
    tsne_prob_dist = np.clip(tsne_prob_dist, 1e-9, None)

    kl_divergence = entropy(original_prob_dist, tsne_prob_dist)
    return tsne_results, kl_divergence

perplexity = 70
learning_rate = 500
tsne_results2, kl_divergence2 = tsne_kl_divergence(similarity_matrix2, perplexity, learning_rate)

stock_ticker = stock_upjong_info['종목코드']
categories_stock = stock_upjong_info['종목명']
industries = stock_upjong_info['업종명']
unique_industries = industries.unique()
num_industries = len(unique_industries)
colors = px.colors.qualitative.Plotly
industry_color_map = {industry: colors[i % len(colors)] for i, industry in enumerate(unique_industries)}

# TSNE 결과를 데이터프레임 형식으로 저장
tsne_df = pd.DataFrame({
    '종목코드': stock_ticker,
    '종목명': categories_stock,
    '업종명': industries,
    'x-axis': tsne_results2[:, 0],
    'y-axis': tsne_results2[:, 1]
})

# 전체 업종의 시가총액 변화 추이 데이터를 준비합니다.
total_agg_data = pd.read_csv('https://raw.githubusercontent.com/pinkocto/DataHub/main/data/tot_market_cap_240327_240524.csv')  # 로컬 파일 경로

total_agg_data['날짜'] = pd.to_datetime(total_agg_data['날짜'])

# Dash 애플리케이션 초기화
app = dash.Dash(__name__)
app.title = "Industry Visualization Dashboard"

app.layout = html.Div(style={'backgroundColor': '#f9f9f9', 'padding': '20px'}, children=[
    html.H1('Fin Visualization', style={'textAlign': 'center', 'color': '#333'}),
    
    # 전체 업종의 시가총액 변화 추이 그래프 추가
    html.Div([
        dcc.Graph(id='market-cap-trend')
    ], style={'marginBottom': '40px'}),  # 그래프와 아래 내용 사이에 여백 추가

    html.Hr(style={'margin': '40px 0'}),  # 위와 아래를 구분하는 수평선

    html.Div([
        html.Div([
            dcc.Graph(id='tsne-scatter'),
            html.Div(id='upjong-info-table')
        ], style={'width': '48%', 'display': 'inline-block', 'padding': '0 20px', 'vertical-align': 'top'}),

        html.Div([
            dcc.Dropdown(
                id='return-frequency',
                options=[
                    {'label': 'Daily', 'value': 'daily'},
                    {'label': 'Weekly', 'value': 'weekly'},
                    {'label': 'Monthly', 'value': 'monthly'},
                    {'label': 'Quarterly', 'value': 'quarterly'}
                ],
                value='daily',
                style={'margin-bottom': '20px'}
            ),
            dcc.Graph(id='volatility-plot'),
            dcc.Graph(id='risk-return-plot')
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'})
    ], style={'display': 'flex', 'justify-content': 'space-between'}),

    html.Hr(style={'margin': '40px 0'}),  # 위와 아래를 구분하는 수평선

    html.Div([
        # html.H2(f'TSNE Results (Perplexity: {perplexity}, Learning Rate: {learning_rate}, KL Divergence: {kl_divergence2:.2f})', style={'textAlign': 'center'}),
        html.H2(''),
        html.Div([
            dcc.Graph(id='tsne-scatter-stock', style={'height': '70vh'}),
        ], style={'marginBottom': '100px'}),  # 그래프와 테이블 사이에 여백 추가
        html.Div([
            dash_table.DataTable(
                id='tsne-stock-table',
                columns=[{"name": i, "id": i} for i in tsne_df.columns],
                data=tsne_df.to_dict('records'),
                style_table={'height': '300px', 'overflowY': 'auto'},
                style_cell={'textAlign': 'left'},
                page_size=10,
                row_selectable='multi',  # 여러 행 선택 가능하도록 설정
                filter_action='native'  # 검색 기능 추가
            )
        ])
    ], style={'padding': '20px'})
])

@app.callback(
    Output('market-cap-trend', 'figure'),
    Input('market-cap-trend', 'relayoutData')
)
def update_market_cap_trend(relayoutData):
    # 전체 업종에 대한 시가총액 변화 추이 그래프
    upjong_name = total_agg_data['업종명'].unique().tolist()  # 전체업종

    # 서버에 저장된 처음 날짜, 마지막 날짜
    start_date = total_agg_data['날짜'].min()
    end_date = total_agg_data['날짜'].max()

    palette = sns.color_palette("hls", len(upjong_name))

    fig = go.Figure()

    # 모든 업종에 대해 시가총액 합계 시계열 그래프 그리기
    for i, industry in enumerate(upjong_name):
        industry_data = total_agg_data[total_agg_data['업종명'] == industry].copy()

        # 첫 번째 값으로 정규화
        industry_data['normalized_market_cap'] = industry_data['tot_market_cap'] / industry_data['tot_market_cap'].iloc[0]

        fig.add_trace(go.Scatter(
            x=industry_data['날짜'], 
            y=industry_data['normalized_market_cap'], 
            mode='lines+markers',
            name=industry,
            line=dict(color=f'rgb({palette[i][0]*255},{palette[i][1]*255},{palette[i][2]*255})')  # convert to rgb string
        ))

    # 그래프 꾸미기
    fig.update_layout(
        title=f'전체 업종의 시가총액의 변화 추이 ({start_date.date()} ~ {end_date.date()})',
        xaxis_title='날짜',
        # yaxis_title='정규화된 시가총액 합계',
        legend_title='업종명',
        legend=dict(x=1.05, y=1, xanchor='left'),  # 범례의 위치 조정
        xaxis=dict(tickformat='%Y-%m-%d')
    )

    return fig

@app.callback(
    Output('tsne-scatter', 'figure'),
    Output('volatility-plot', 'figure'),
    Output('risk-return-plot', 'figure'),
    Output('upjong-info-table', 'children'),
    Input('tsne-scatter', 'hoverData'),
    Input('return-frequency', 'value')
)
def update_graphs(hoverData, frequency):
    # TSNE Plot
    categories = daily_simple_returns.columns
    tsne_fig = px.scatter(
        x=tsne_results[:, 0], 
        y=tsne_results[:, 1], 
        text=categories,
        labels={'x': 'TSNE 1', 'y': 'TSNE 2'},
        title="Visualization of Industry Similarities"
    )
    tsne_fig.update_layout(plot_bgcolor='white', paper_bgcolor='white')

    # 주기별 수익률 선택
    if frequency == 'daily':
        selected_returns = daily_simple_returns
    elif frequency == 'weekly':
        selected_returns = weekly_returns
    elif frequency == 'monthly':
        selected_returns = monthly_returns
    elif frequency == 'quarterly':
        selected_returns = quarterly_returns

    # Volatility Plot
    vol_fig = go.Figure()
    if hoverData:
        industry = hoverData['points'][0]['text']
        vol_data = selected_returns[industry]
        vol_fig.add_trace(go.Scatter(x=vol_data.index, y=vol_data, mode='lines', name=industry))

    vol_fig.update_layout(
        title='Volatility',
        xaxis_title='Date',
        yaxis_title='Returns',
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # Risk vs. Expected Return Plot
    rets = selected_returns.dropna()
    text = list(rets.mean().index.values)
    risk_return_fig = go.Figure()

    risk_return_fig.add_trace(go.Scatter(
        x=rets.mean(),
        y=rets.std(),
        mode='markers+text',
        text=text,
        textposition='top center',
        marker=dict(size=10, color='darkorange', line=dict(color='darkorange', width=1))
    ))

    if hoverData:
        industry = hoverData['points'][0]['text']
        mean_return = rets.mean()[industry]
        std_risk = rets.std()[industry]
        risk_return_fig.add_trace(go.Scatter(
            x=[mean_return],
            y=[std_risk],
            mode='markers+text',
            text=[industry],
            textposition='top center',
            marker=dict(size=15, color='red', line=dict(color='red', width=2))
        ))

    # xaxis 및 yaxis 범위를 동적으로 설정
    xaxis_range = [rets.mean().min() * 1.1, rets.mean().max() * 1.1]
    yaxis_range = [0, rets.std().max() * 1.1]

    risk_return_fig.update_layout(
        title="Risk vs. Expected Return",
        xaxis=dict(title="Expected Return", range=xaxis_range),
        yaxis=dict(title="Standard Deviation (Risk)", range=yaxis_range),
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False
    )

    # 업종 정보 테이블 업데이트
    if hoverData:
        industry = hoverData['points'][0]['text']
        filtered_data = stock_upjong_info[stock_upjong_info['업종명'] == industry]
        table = dash_table.DataTable(
            columns=[{"name": i, "id": i} for i in filtered_data.columns],
            data=filtered_data.to_dict('records'),
            style_table={'height': '300px', 'overflowY': 'auto'}
        )
    else:
        table = html.Div("Hover over a point to see details")

    return tsne_fig, vol_fig, risk_return_fig, table

@app.callback(
    Output('tsne-scatter-stock', 'figure'),
    Input('tsne-stock-table', 'selected_rows'),
    State('tsne-stock-table', 'data')
)
def update_tsne_scatter_stock(selected_rows, table_data):
    tsne_fig2 = go.Figure()

    unique_legend_added = set()  # 추가된 범례 항목을 추적하기 위한 집합
    
    if selected_rows:
        for i, row_index in enumerate(selected_rows):
            row = table_data[row_index]
            stock, industry = row['종목명'], row['업종명']
            x, y = row['x-axis'], row['y-axis']
            color = industry_color_map[industry]
            tsne_fig2.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                text=[stock],
                textposition='top center',
                marker=dict(size=15, color='red', line=dict(color=color, width=1)),
                name=industry if industry not in unique_legend_added else None,
                showlegend=industry not in unique_legend_added
            ))
            unique_legend_added.add(industry)

        for i, row in enumerate(table_data):
            if i not in selected_rows:
                stock, industry = row['종목명'], row['업종명']
                x, y = row['x-axis'], row['y-axis']
                color = industry_color_map[industry]
                tsne_fig2.add_trace(go.Scatter(
                    x=[x],
                    y=[y],
                    mode='markers',
                    marker=dict(size=10, color=color, opacity=0.1, line=dict(color=color, width=1)),
                    name=industry if industry not in unique_legend_added else None,
                    showlegend=industry not in unique_legend_added
                ))
                unique_legend_added.add(industry)
    else:
        for i, (stock, industry) in enumerate(zip(categories_stock, industries)):
            color = industry_color_map[industry]
            tsne_fig2.add_trace(go.Scatter(
                x=[tsne_results2[i, 0]],
                y=[tsne_results2[i, 1]],
                mode='markers+text',
                text=[stock],
                textposition='top center',
                marker=dict(size=10, color=color, line=dict(color=color, width=1)),
                name=industry if industry not in unique_legend_added else None,
                showlegend=industry not in unique_legend_added,
                opacity=1.0
            ))
            unique_legend_added.add(industry)

    tsne_fig2.update_layout(
        # title=f'TSNE Results (Perplexity: {perplexity}, Learning Rate: {learning_rate}, KL Divergence: {kl_divergence2:.2f})',
        # xaxis=dict(title="TSNE 1"),
        # yaxis=dict(title="TSNE 2"),
        height=800,  # 높이를 조정하여 더 큰 그래프 표시
        showlegend=True,
        legend=dict(itemsizing='constant')
    )

    return tsne_fig2

# if __name__ == '__main__':
#     app.run_server(debug=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7996)