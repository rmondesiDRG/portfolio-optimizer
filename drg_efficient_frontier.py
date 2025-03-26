import streamlit as st
import datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import scipy.optimize as sc
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="Portfolio Optimizer", page_icon="📈")
st.title("📊 Portfolio Optimizer with Efficient Frontier")

st.sidebar.header("User Input Parameters")
stocks = st.sidebar.text_input("Enter tickers (comma separated):", "AAPL,MSFT,GOOGL,NVDA,XPEV,ETN").upper().split(',')
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 10.0, 3.0) / 100
constraint_set = (0.0, 1.0)
days_back = st.sidebar.slider("Lookback Period (days):", 90, 1095, 365)
target_return_input = st.sidebar.text_input("Target Returns (comma-separated decimals):", "0.08, 0.12, 0.15")

end_date = dt.datetime.now()
start_date = end_date - dt.timedelta(days=days_back)

def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end)['Close']
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

def portfolioPerformance(weights, meanReturns, covMatrix):
    annualReturn = np.sum(meanReturns * weights) * 252
    annualVolatility = np.sqrt(np.dot(weights, np.dot(covMatrix, weights))) * np.sqrt(252)
    return annualReturn, annualVolatility

def negativeSR(weights, meanReturns, covMatrix, riskFreeRate=0.0):
    pReturn, pVol = portfolioPerformance(weights, meanReturns, covMatrix)
    return -(pReturn - riskFreeRate) / pVol

def maxSR(meanReturns, covMatrix, riskFreeRate=0.0, constrainSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple(constrainSet for _ in range(numAssets))
    result = sc.minimize(negativeSR, [1. / numAssets] * numAssets, args=args, method="SLSQP", bounds=bounds, constraints=constraints)
    return result

def minimizeVariance(meanReturns, covMatrix, constrainSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},)
    bounds = tuple(constrainSet for _ in range(numAssets))
    result = sc.minimize(lambda w, meanR, cov: portfolioPerformance(w, meanR, cov)[1], [1. / numAssets] * numAssets, args=args, method="SLSQP", bounds=bounds, constraints=constraints)
    return result

def efficientFrontier(meanReturns, covMatrix, returnTarget, constrainSet=(0, 1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = (
        {'type': 'eq', 'fun': lambda x: portfolioPerformance(x, meanReturns, covMatrix)[0] - returnTarget},
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    )
    bounds = tuple(constrainSet for _ in range(numAssets))
    effOpt = sc.minimize(lambda w, meanR, cov: portfolioPerformance(w, meanR, cov)[1], [1. / numAssets] * numAssets, args=args, method="SLSQP", bounds=bounds, constraints=constraints)
    return effOpt

def randomPortfolios(meanReturns, covMatrix, numPortfolios=5000, riskFreeRate=0.0):
    numAssets = len(meanReturns)
    results = np.zeros((3, numPortfolios))
    for i in range(numPortfolios):
        weights = np.random.rand(numAssets)
        weights /= np.sum(weights)
        pRet, pVol = portfolioPerformance(weights, meanReturns, covMatrix)
        results[0, i] = pRet
        results[1, i] = pVol
        results[2, i] = (pRet - riskFreeRate) / pVol
    return results, None

def capitalMarketLine(riskFreeRate, maxSR_return, maxSR_vol, extendRatio=1.2, nPoints=100):
    slope = (maxSR_return - riskFreeRate) / maxSR_vol
    volRange = np.linspace(0, maxSR_vol * extendRatio, nPoints)
    retRange = riskFreeRate + slope * volRange
    return volRange, retRange

def calculatedResults(meanReturns, covMatrix, riskFreeRate=0.0, constraintSet=(0, 1)):
    maxSR_Portfolio = maxSR(meanReturns, covMatrix, riskFreeRate, constraintSet)
    maxSR_Return, maxSR_Volatility = portfolioPerformance(maxSR_Portfolio.x, meanReturns, covMatrix)
    maxSR_allocation = pd.Series(maxSR_Portfolio.x, index=meanReturns.index) * 100
    minVar_Portfolio = minimizeVariance(meanReturns, covMatrix, constraintSet)
    minVar_Return, minVar_Volatility = portfolioPerformance(minVar_Portfolio.x, meanReturns, covMatrix)
    minVar_allocation = pd.Series(minVar_Portfolio.x, index=meanReturns.index) * 100
    targetReturns = np.linspace(minVar_Return, maxSR_Return, 20)
    frontierVols = [efficientFrontier(meanReturns, covMatrix, r, constraintSet).fun for r in targetReturns]
    return maxSR_Return, maxSR_Volatility, maxSR_allocation, minVar_Return, minVar_Volatility, minVar_allocation, None, frontierVols, targetReturns

def EF_graph(meanReturns, covMatrix, riskFreeRate=0.0, constraintSet=(0, 1), numPortfolios=5000):
    maxSR_Return, maxSR_Volatility, maxSR_allocation, \
    minVar_Return, minVar_Volatility, minVar_allocation, \
    frontierWeights, frontierVols, targetReturns = calculatedResults(
        meanReturns, covMatrix, riskFreeRate, constraintSet
    )

    randomResults, _ = randomPortfolios(meanReturns, covMatrix, numPortfolios, riskFreeRate)
    volCML, retCML = capitalMarketLine(riskFreeRate, maxSR_Return, maxSR_Volatility, extendRatio=1.5)

    data = [
        go.Scatter(x=randomResults[1] * 100, y=randomResults[0] * 100, mode='markers',
                   marker=dict(color='rgba(0,0,255,0.3)', size=5), name='Random Portfolios'),
        go.Scatter(x=[minVar_Volatility * 100], y=[minVar_Return * 100], mode='markers',
                   marker=dict(size=10, color='green', line=dict(width=2, color='black')), name='Min Variance'),
        go.Scatter(x=[maxSR_Volatility * 100], y=[maxSR_Return * 100], mode='markers',
                   marker=dict(size=10, color='red', line=dict(width=2, color='black')), name='Max Sharpe (Tangency)'),
        go.Scatter(x=[v * 100 for v in frontierVols], y=[r * 100 for r in targetReturns], mode='lines',
                   line=dict(color='black', width=2, dash='dash'), name='Efficient Frontier'),
        go.Scatter(x=volCML * 100, y=retCML * 100, mode='lines',
                   line=dict(color='purple', width=2), name='Capital Market Line')
    ]

    layout = go.Layout(
        title=f"Efficient Frontier & CML (Risk-free = {riskFreeRate * 100:.2f}%)",
        xaxis=dict(title="Annualized Volatility (%)"),
        yaxis=dict(title="Annualized Return (%)"),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(240,240,240,0.7)'),
        width=900,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='whitesmoke'
    )
    fig = go.Figure(data=data, layout=layout)
    st.plotly_chart(fig, use_container_width=True)

meanReturns, covMatrix = getData(stocks, start_date, end_date)
EF_graph(meanReturns, covMatrix, riskFreeRate=risk_free_rate, constraintSet=constraint_set)

st.markdown("""
### 📈 Graph Overview
This chart visualizes the efficient frontier, random portfolios, the tangency (max Sharpe) portfolio, minimum variance portfolio, and the capital market line. It helps assess risk-return tradeoffs.
""")

maxSR_Return, maxSR_Volatility, maxSR_allocation, minVar_Return, minVar_Volatility, minVar_allocation, _, _, _ = calculatedResults(meanReturns, covMatrix, risk_free_rate, constraint_set)

st.markdown("### 🟥 Max Sharpe Portfolio Allocation")
st.dataframe(maxSR_allocation.to_frame(name="allocation").style.format("{:.2f}").set_caption("Max Sharpe Portfolio"))

st.markdown("### 🟩 Minimum Variance Portfolio Allocation")
st.dataframe(minVar_allocation.to_frame(name="allocation").style.format("{:.2f}").set_caption("Minimum Variance Portfolio"))

try:
    desired_returns = [float(r.strip()) for r in target_return_input.split(",")]

    def weights_CML_return(desired_returns, maxSR_Return, maxSR_Volatility, maxSR_allocation, riskFreeRate):
        data = {}
        feedback = []
        for desired_return in desired_returns:
            desired_weight = (desired_return - riskFreeRate) / (maxSR_Return - riskFreeRate)
            desired_weight_rf = 1 - desired_weight
            desired_volatility = desired_weight * maxSR_Volatility
            desired_asset_allocations = (maxSR_allocation / 100) * desired_weight * 100
            label = f"Target Return: {desired_return * 100:.2f}%"
            data[label] = desired_asset_allocations.round(2)

            comment = f"📌 Portfolio Allocation for {label}\n→  Portfolio Allocation for {label} \n→ Portfolio Volatility: {desired_volatility * 100:.2f}%"
            if desired_weight_rf >= 0:
                comment += f"\n→ Weight in Risk-Free Asset: {desired_weight_rf * 100:.2f}%"
            else:
                comment += f"\n→ Leverage Required: Borrow {-desired_weight_rf * 100:.2f}% at Risk-Free Rate"
            comment += f"\n→ Weight in Tangency Portfolio: {desired_weight * 100:.2f}%"
            feedback.append(comment)

        df = pd.DataFrame(data)
        df.index.name = "Asset"
        return df, feedback

    cml_df, feedback = weights_CML_return(desired_returns, maxSR_Return, maxSR_Volatility, maxSR_allocation, risk_free_rate)

    st.markdown("### 💹 Capital Market Line Allocations")
    st.dataframe(cml_df.style.set_caption("Asset Allocation for Target Returns (CML)").format("{:.2f}"))

    for note in feedback:
        st.markdown(f"```{note}```")

except:
    st.warning("Invalid input in target return field. Please enter valid decimals.")
