import yfinance as yf
import pickle
import datetime as dt
import pandas as pd
import streamlit as st
from PIL import Image
from prophet.plot import plot_plotly
from plotly import graph_objs as go


crypto_symbols = ['BTC', 'ETH', 'LTC', 'DOGE', 'SOL', 'USDT', 'USDC', 'BNB', 'XRP', 'ADA', 'DAI', 'WTRX',
                  'DOT', 'HEX', 'TRX', 'SHIB', 'LEO', 'WBTC', 'AVAX', 'YOUC', 'MATIC', 'UNI1', 'STETH', 'LTC', 'FTT']

# load data from yfinance
def load_data(symbol):
    yf_data = yf.Ticker(f'{symbol}-USD').history(start='2014-01-01', end=dt.datetime.now(), interval='1d')
    yf_data.reset_index(inplace=True)
    yf_data.drop(['Dividends', 'Stock Splits'], axis='columns', inplace=True)
    return yf_data
st.set_page_config(layout='wide')

st.sidebar.write('author: Bonaventure Osuide')

image1 = Image.open('My logo.jpeg')
st.sidebar.image(image1, width=300)
selected_coins = st.sidebar.selectbox("Select Coin Interest", crypto_symbols)
intervals = ['days', 'months', 'years']
selected_interval = st.sidebar.selectbox('Select Interval measurement', intervals)
interval = int(st.sidebar.number_input('Input Interval'))
if selected_interval == 'days':
    new_interval = interval
elif selected_interval == 'months':
    new_interval = interval * 31
else:
    new_interval = interval * 365

with st.spinner('Load data...'):
    data = load_data(selected_coins)

st.title('INTELLIGENT COIN TRADING SYSTEM')
st.markdown('''This app gives an estimate of the price of a given currency in the future and shows you dates you can 
purchase a coin to give you a profit''')
st.success('Sucessfully loaded the data!')

st.subheader('Correlation of 10 Coins')
st.markdown('''The correlation shows coins that are likely to have the same trend in the market. Coins having correlation
 of about 0.5 are correlated(both +ve and -ve) while those having above 0.7 have very strong correlation correlated''')
image = Image.open('download.png')
st.image(image, width=600)


def visualize_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close price'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data.Close.rolling(30).mean(), name='Moving average'))
    fig.update_layout(paper_bgcolor='lightgrey', title_text=f'Time Series and Moving Average of {selected_coins}', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

visualize_raw_data()

# Frecasting

df_train = data[['Date', 'Close']]


df_train = df_train.rename(columns={'Date':"ds", "Close": 'y'})

pickled_model = pickle.load(open('model.pkl', 'rb'))
future = pickled_model.make_future_dataframe(periods=new_interval)
prediction = pickled_model.predict(future)

st.subheader('Forecast data')
st.write(prediction.rename(columns={'ds':'date','yhat':'predicted prices'}).tail()[['date','predicted prices']])

# getting the present price, future price and future date of the coin
present_price = list(data.Close)[-1]
future_price = list(prediction.yhat)[-1]
future_date = list(prediction.ds)[-1]
st.subheader('Forecast Profit')
st.write(f'Today\'s Price of {selected_coins}: $', format(round(present_price, 2)))
if selected_interval == 'days':
    st.write(f'Forcasted price of {selected_coins} in {interval} days is $', format(round(future_price, 2)))
elif selected_interval == 'months':
    if interval == 1:
        st.write(f'Tomorrows price of {selected_coins} is {interval} month is $', format(round(future_price, 2)))
    else:
        st.write(f'Forcasted price of {selected_coins} in {interval} months is $', format(round(future_price, 2)))

else:
    if interval == 1:
        st.write(f'Forcasted price of {selected_coins} in {interval} year is $', format(round(future_price, 2)))
    else:
        st.write(f'Forcasted price of {selected_coins} in {interval} years is $', format(round(future_price, 2)))
value = future_price - present_price
if future_price > present_price:
    st.write('You are likely to make a profit of about: $', format(round(value, 2)), f'in {future_date}')
else:
    st.write('You are likely to run at a loss of about: $', format(abs(round(value, 2))), f'in {future_date}')


st.subheader('Visualization of Forecast data')
fig1 = plot_plotly(pickled_model, prediction)
st.plotly_chart(fig1)

st.subheader('Forecast Components')
fig2 = pickled_model.plot_components(prediction)
st.write(fig2)

st.subheader('Forecast Price with Date')

date = st.date_input(f'Forecast the future price of {selected_coins} by selecting the date',
                    dt.datetime.now())

future_date = pd.DataFrame({'ds':[date]})

predicted_price = pickled_model.predict(future_date)

st.write(f'The predicted price of {selected_coins} in {date} is $',format(round(predicted_price.iloc[0,-1], 2)))
def plot():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close price'))
    fig.add_trace(go.Scatter(x=predicted_price['ds'], y=predicted_price.yhat, name='Predicted price'))
    fig.update_layout(paper_bgcolor='lightblue', title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot()