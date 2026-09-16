import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ----------------------------
# 기본 설정
# ----------------------------
st.set_page_config(
    page_title="서울 100년 기온 변화",
    page_icon="🌡️",
    layout="wide",
)

DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"


@st.cache_data(show_spinner="서울 기온 데이터를 불러오는 중...")
def load_data(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, encoding="utf-8-sig")
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])
    df["연도"] = df["날짜"].dt.year
    return df


@st.cache_data(show_spinner=False)
def compute_yearly(df: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            연평균최저기온=("최저기온", "mean"),
            연평균최고기온=("최고기온", "mean"),
            관측일수=("평균기온", "count"),
        )
        .reset_index()
    )
    # 관측치가 너무 적은 해(첫해·마지막해 등)는 제외해 왜곡을 줄인다
    yearly = yearly[yearly["관측일수"] >= 300].reset_index(drop=True)
    return yearly


# ----------------------------
# 데이터 로드
# ----------------------------
raw_df = load_data(DATA_URL)
yearly_df = compute_yearly(raw_df)

# ----------------------------
# 헤더
# ----------------------------
st.title("🌡️ 서울, 100년의 기온 변화")
st.caption(
    f"서울 기상 관측 데이터({int(yearly_df['연도'].min())}년 ~ {int(yearly_df['연도'].max())}년) 기반 연평균 기온 추이"
)

# ----------------------------
# 사이드바 - 옵션
# ----------------------------
st.sidebar.header("⚙️ 옵션")

min_year, max_year = int(yearly_df["연도"].min()), int(yearly_df["연도"].max())
year_range = st.sidebar.slider(
    "표시할 연도 범위",
    min_value=min_year,
    max_value=max_year,
    value=(min_year, max_year),
)

show_minmax = st.sidebar.checkbox("최저·최고 기온 평균도 함께 보기", value=False)
show_trend = st.sidebar.checkbox("추세선(선형회귀) 표시", value=True)
show_ma = st.sidebar.checkbox("10년 이동평균 표시", value=True)

filtered = yearly_df[
    (yearly_df["연도"] >= year_range[0]) & (yearly_df["연도"] <= year_range[1])
].copy()

# ----------------------------
# 핵심 지표
# ----------------------------
col1, col2, col3 = st.columns(3)

first_val = filtered["연평균기온"].iloc[0]
last_val = filtered["연평균기온"].iloc[-1]
diff = last_val - first_val

with col1:
    st.metric(
        f"{filtered['연도'].iloc[0]}년 연평균기온",
        f"{first_val:.1f} °C",
    )
with col2:
    st.metric(
        f"{filtered['연도'].iloc[-1]}년 연평균기온",
        f"{last_val:.1f} °C",
    )
with col3:
    st.metric(
        "기간 내 변화량",
        f"{diff:+.1f} °C",
        delta=f"{diff:+.1f} °C",
    )

# ----------------------------
# 메인 그래프
# ----------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=filtered["연도"],
        y=filtered["연평균기온"],
        mode="lines+markers",
        name="연평균기온",
        line=dict(color="#e74c3c", width=2),
        marker=dict(size=4),
    )
)

if show_minmax:
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=filtered["연평균최고기온"],
            mode="lines",
            name="연평균 최고기온",
            line=dict(color="#f39c12", width=1.5, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=filtered["연평균최저기온"],
            mode="lines",
            name="연평균 최저기온",
            line=dict(color="#3498db", width=1.5, dash="dot"),
        )
    )

if show_ma and len(filtered) >= 10:
    filtered["10년이동평균"] = filtered["연평균기온"].rolling(window=10, min_periods=5).mean()
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=filtered["10년이동평균"],
            mode="lines",
            name="10년 이동평균",
            line=dict(color="#2c3e50", width=3),
        )
    )

if show_trend and len(filtered) >= 2:
    coef = np.polyfit(filtered["연도"], filtered["연평균기온"], 1)
    trend_y = np.polyval(coef, filtered["연도"])
    fig.add_trace(
        go.Scatter(
            x=filtered["연도"],
            y=trend_y,
            mode="lines",
            name=f"추세선 (연 {coef[0]:+.3f}°C)",
            line=dict(color="gray", width=2, dash="dash"),
        )
    )
    st.sidebar.markdown(
        f"**추세선 기울기:** 연평균 약 `{coef[0]:+.3f}°C`씩 변화\n\n"
        f"(10년당 약 `{coef[0]*10:+.2f}°C`)"
    )

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=10, r=10, t=30, b=10),
    height=550,
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# 데이터 표 (선택적으로 펼쳐보기)
# ----------------------------
with st.expander("📋 연도별 데이터 표 보기"):
    st.dataframe(
        filtered[["연도", "연평균기온", "연평균최저기온", "연평균최고기온", "관측일수"]]
        .sort_values("연도", ascending=False)
        .style.format(
            {
                "연평균기온": "{:.1f}",
                "연평균최저기온": "{:.1f}",
                "연평균최고기온": "{:.1f}",
            }
        ),
        use_container_width=True,
    )

st.caption("데이터 출처: greatsong/modudata (기상청 서울 관측소, 지점번호 108)")
