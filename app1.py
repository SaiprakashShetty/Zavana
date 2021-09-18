import numpy as np
from numpy.lib.function_base import append
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from flask import Flask,flash  
from datetime import datetime
from flask import render_template  
from flask import request, redirect, url_for      # import flask
from sklearn import preprocessing

#import fundamental
app = Flask(__name__)             # create an app instance
app.static_folder = 'static'
app.secret_key = 'random string'

@app.route("/")                   # at the end point /
def mainpage():                      # call method hello
    return render_template("index.html") 

@app.route('/predictnow', methods=['GET', 'POST'])
def predictnow():
    if request.method == 'POST':
        return redirect(url_for('mainpage'))
    return render_template('inner-page.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        ticker = request.form.get("ticker")
        stock = yf.Ticker(ticker+".NS")
        
        try:
            stock = yf.Ticker(ticker+".NS")
            try:
                value=stock.info['longBusinessSummary']
            except KeyError:
                value="-"
            fullname=stock.info['longName']
            website=stock.info['website']
            bookvalue=stock.info['bookValue']
            fiftytwochange=stock.info['52WeekChange']
            beta=stock.info['beta']
            currentPrice=stock.info['currentPrice']
            dayHigh=stock.info['dayHigh']
            dayLow=stock.info['dayLow']
            dividendRate=stock.info['dividendRate']
            fiftyTwoWeekHigh=stock.info['fiftyTwoWeekHigh']
            fiftyTwoWeekLow=stock.info['fiftyTwoWeekLow']
            forwardEps=stock.info['forwardEps']
            forwardPE=stock.info['forwardPE']
            marketCap=stock.info['marketCap']
            priceToBook=stock.info['priceToBook']
            returnOnAssets=stock.info['returnOnAssets']
            revenueGrowth=stock.info['revenueGrowth']
            targetHighPrice=stock.info['targetHighPrice']
            targetLowPrice=stock.info['targetLowPrice']
            trailingEps=stock.info['trailingEps']
            try:
                trailingPE=stock.info['trailingPE']
            except KeyError:
                trailingPE="-"
            returnOnEquity=stock.info['returnOnEquity']
            twoHundredDayAverage=stock.info['twoHundredDayAverage']
            previousclose=stock.info['previousClose']
        except KeyError:
            return render_template('ErrorPage.html')

        
        ###Prediction Start
        regressor = load_model(ticker+".h5")
        # Last 10 days prices
        #stock = yf.Ticker("--ticker name--.NS")
        dt = stock.history(interval="1d", period="10d")
        new_dt = dt.filter(['Close'])
        sc=MinMaxScaler()
        DataScaler = sc.fit(new_dt)
        last10Days = new_dt[-10:].values
        Last10Days = []
        ##Append the past 10 days
        Last10Days.append(last10Days)

        ##Converting the X_test_data into a numpy array
        Last10Days = np.array(Last10Days)
 
        # Normalizing the data just like we did for training the model
        Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))
 
        # Changing the shape of the data to 3D
        # Choosing TimeSteps as 10 because we have used the same for training
        NumSamples=1
        TimeSteps=10
        NumFeatures=1
        Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)

        # Making predictions on data
        predicted_Price = regressor.predict(Last10Days)
        predicted_Price = DataScaler.inverse_transform(predicted_Price)
        tom_price=predicted_Price[0][0]
        changefromtoda = round(tom_price-currentPrice, 3)
        changefromtodaype = round(((tom_price-currentPrice)/currentPrice)*100,3)
        changefromtoday=""
        changefromtodayper=""
        predsign=""
        if(changefromtoda>0):
            changefromtoday= "+"+str(changefromtoda)
            changefromtodayper="+"+str(changefromtodaype)
            predsign="+"
        else :
            changefromtoday=changefromtoda
            changefromtodayper=changefromtodaype
            predsign="-"
        #print(predicted_Price)
        ####Prediction End
        
        changefromyesterda = round(currentPrice-previousclose, 3)
        changefromyesterdaype = round(((currentPrice-previousclose)/previousclose)*100,3)
        changefromyesterday=""
        changefromyesterdayper=""
        sign=""
        if(changefromyesterda>0):
            changefromyesterday= "+"+str(changefromyesterda)
            changefromyesterdayper="+"+str(changefromyesterdaype)
            sign="+"
        else :
            changefromyesterday=changefromyesterda
            changefromyesterdayper=changefromyesterdaype
            sign="-"
        req = request.form
        print(req)

        d = stock.history(period="1y",interval="1d")
        d.drop(['Dividends','Stock Splits'], axis=1, inplace=True)
        d['Returns']=((d['Close']-d['Open'])/d['Open'])*100
        d.to_csv('Stock_data.csv')
        df = pd.read_csv('Stock_data.csv')
        df.sort_values('Date')
        arr = df.to_numpy()

        stockDate=[]
        closePrice=[]
        highPrice=[]
        lowPrice=[]
        openPrice=[]   
        dfnorm=d
        x = dfnorm.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        dfnorm = pd.DataFrame(x_scaled)
        dfnorm['Date']=df['Date']
        dfnorm.to_csv('Stock_datanorm.csv')
        df2 = pd.read_csv('Stock_datanorm.csv')
        df2.sort_values('Date')
        arr2 = df2.to_numpy()
        volume=[]
        Returns=[]
        i=0
        for row in arr:
            stockDate.append(arr[i][0])
            closePrice.append(arr[i][4])
            highPrice.append(arr[i][2])
            lowPrice.append(arr[i][3])
            openPrice.append(arr[i][1])
            i=i+1
        j=0
        for row in arr2:
            volume.append(arr2[j][4])
            Returns.append(arr2[j][5])
            j=j+1
        
        #Recommendation
        temp = stock.history(period="2y",interval="1d")
        temp.drop(['Dividends','Stock Splits'], axis=1, inplace=True)
        temp['Returns']=((temp['Close']-temp['Open'])/temp['Open'])*100
        temp.to_csv('Recommendation.csv')
        stock_data = pd.read_csv('Recommendation.csv')
        stock_data.sort_values('Date')

        stock_data['SMA_25'] = stock_data['Close'].rolling(25).mean()
        stock_data['SMA_50'] = stock_data['Close'].rolling(50).mean()
        stock_data['Signal_SMA'] = np.where(stock_data['SMA_25'] > stock_data['SMA_50'], 1.0, 0.0)
        stock_data['Position_SMA'] = stock_data['Signal_SMA'].diff()
        stock_data = stock_data.dropna()

        stock_data['EMA_25'] = stock_data['Close'].ewm(span= 25, adjust=False).mean()
        stock_data['EMA_50'] = stock_data['Close'].ewm(span= 50, adjust=False).mean()
        stock_data['Signal_EMA'] = np.where(stock_data['EMA_25'] > stock_data['EMA_50'], 1.0, 0.0)
        stock_data['Position_EMA'] = stock_data['Signal_EMA'].diff()

        stock_data['Rolling Mean'] = stock_data['Close'].rolling(20).mean()
        stock_data['Rolling Std'] = stock_data['Close'].rolling(20).std()
        stock_data['Bollinger High'] = stock_data['Rolling Mean'] + (stock_data['Rolling Std']*2) 
        stock_data['Bollinger Low'] = stock_data['Rolling Mean'] - (stock_data['Rolling Std']*2)
        stock_data = stock_data.dropna()
        stock_data['Signal_BB'] = None
        stock_data['Position_BB'] = None
        for row in range(len(stock_data)):
            if (stock_data['Close'].iloc[row] > stock_data['Bollinger High'].iloc[row]) and (stock_data['Close'].iloc[row-1] < stock_data['Bollinger High'].iloc[row-1]):stock_data['Signal_BB'].iloc[row] = 0
            if (stock_data['Close'].iloc[row] < stock_data['Bollinger Low'].iloc[row]) and (stock_data['Close'].iloc[row-1] > stock_data['Bollinger Low'].iloc[row-1]):stock_data['Signal_BB'].iloc[row] = 1  

        stock_data['Signal_BB'].fillna(method='ffill',inplace=True)
        stock_data['Position_BB'] = stock_data['Signal_BB'].diff()

        stock_data['MACD'] = stock_data['Close'].ewm(span=12, adjust= False).mean() - stock_data['Close'].ewm(span=26, adjust= False).mean()
        stock_data['Signal_9'] = stock_data['MACD'].ewm(span=9, adjust= False).mean()
        stock_data['Signal_MACD'] = np.where(stock_data.loc[:, 'MACD'] > stock_data.loc[:, 'Signal_9'], 1.0, 0.0)
        stock_data['Position_MACD'] = stock_data['Signal_MACD'].diff()
        
        stock_data['Diff'] = stock_data['Close'].diff()
        stock_data['Gain'] = stock_data['Diff'][stock_data['Diff']>0]
        stock_data['Loss'] = (-1)*stock_data["Diff"][stock_data["Diff"]<0]
        stock_data = stock_data.fillna(0)
        stock_data['AvgGain'] = stock_data['Gain'].rolling(window=14).mean()
        stock_data['AvgLoss'] = stock_data['Loss'].rolling(window=14).mean()
        stock_data['RSI'] = 100 - (100/(1+ stock_data['AvgGain']/stock_data['AvgLoss']))
        stock_data['RSI70'] = np.where(stock_data['RSI'] > 70, 1, 0)
        stock_data['RSI30'] = np.where(stock_data['RSI'] < 30, -1, 0)
        stock_data['Position_R70'] = stock_data['RSI70'].diff()
        stock_data['Position_R30'] = stock_data['RSI30'].diff()

        stock_data = stock_data.dropna()
        #stock_data.to_csv('smacheck.csv')
        arr3=stock_data.to_numpy()
        
        buyx=[]
        buyy=[]
        sellx=[]
        selly=[]
        emabuyx=[]
        emabuyy=[]
        emasellx=[]
        emaselly=[]
        bbhigh=[]
        bblow=[]
        bbmean=[]
        bbbuyx=[]
        bbbuyy=[]
        bbsellx=[]
        bbselly=[]
        sig9=[]
        macd=[]
        cdbuyx=[]
        cdbuyy=[]
        cdsellx=[]
        cdselly=[]
        rsi=[]
        r70=[]
        r30=[]
        rsbuyx=[]
        rsbuyy=[]
        rssellx=[]
        rsselly=[]

        s=0
        for row in arr3:
            if arr3[s][10]==1:
               buyx.append(arr3[s][0])
               buyy.append(arr3[s][8])
            elif arr3[s][10]==-1:
               sellx.append(arr3[s][0])
               selly.append(arr3[s][8])
            s=s+1
        showsmabuydate=buyx[-1]
        showsmabuyprice=round(buyy[-1],3)
        showsmaselldate=sellx[-1]
        showsmasellprice=round(selly[-1],3)

        t=0
        for row in arr3:
            if arr3[t][14]==1:
                emabuyx.append(arr3[t][0])
                emabuyy.append(arr3[t][12])
            elif arr3[t][14]==-1:
               emasellx.append(arr3[t][0])
               emaselly.append(arr3[t][12])
            t=t+1
        showemabuydate=emabuyx[-1]
        showemabuyprice=round(emabuyy[-1],3)
        showemaselldate=emasellx[-1]
        showemasellprice=round(emaselly[-1],3)

        x=0
        for row in arr3:
            if arr3[x][20]==1:
                bbbuyx.append(arr3[x][0])
                bbbuyy.append(arr3[x][18])
            elif arr3[x][20]==-1:
                bbsellx.append(arr3[x][0])
                bbselly.append(arr3[x][17])
            x=x+1
        showbbbuydate=bbbuyx[-1]
        showbbbuyprice=round(bbbuyy[-1],3)
        showbbselldate=bbsellx[-1]
        showbbsellprice=round(bbselly[-1],3)
        
        y=0
        for row in arr3:
            if arr3[y][24]==1:
                cdbuyx.append(arr3[y][0])
                cdbuyy.append(arr3[y][4])
            elif arr3[y][24]==-1:
                cdsellx.append(arr3[y][0])
                cdselly.append(arr3[y][4])
            y=y+1
        showcdbuydate=cdbuyx[-1]
        showcdbuyprice=round(cdbuyy[-1],3)
        showcdselldate=cdsellx[-1]
        showcdsellprice=round(cdselly[-1],3)
        z=0
        for row in arr3:
            if arr3[z][33]==1:
                rsbuyx.append(arr3[z][0])
                rsbuyy.append(arr3[z][30])
            elif arr3[z][34]==-1:
                rssellx.append(arr3[z][0])
                rsselly.append(arr3[z][30])
            z=z+1
        showrsibuydate=rsbuyx[-1]
        showrsibuyprice=round(rsbuyy[-1],3)
        showrsiselldate=rssellx[-1]
        showrsisellprice=round(rsselly[-1],3)

        sma25=[]
        sma50=[]
        possma=[]
        sigsma=[]
        madate=[]
        maclose=[]
        ema25=[]
        ema50=[]

        k=0
        for row in arr3:
            madate.append(arr3[k][0])
            maclose.append(arr3[k][4])
            sma25.append(arr3[k][7])
            sma50.append(arr3[k][8])
            sigsma.append(arr3[k][9])
            possma.append(arr3[k][10])
            ema25.append(arr3[k][11])
            ema50.append(arr3[k][12])
            bbmean.append(arr3[k][15])
            bbhigh.append(arr3[k][17])
            bblow.append(arr3[k][18])
            sig9.append(arr3[k][22])
            macd.append(arr3[k][21])
            rsi.append(arr3[k][30])
            r70.append(70)
            r30.append(30)
            k=k+1
        
        return render_template("prediction_1.html",
        ticker=ticker, value  = value, fullname=fullname, website=website, bookvalue=bookvalue,
        fiftytwochange=fiftytwochange,beta=beta,currentPrice=currentPrice, dayHigh=dayHigh,
        dayLow=dayLow, dividendRate = dividendRate, fiftyTwoWeekHigh=fiftyTwoWeekHigh,
        fiftyTwoWeekLow=fiftyTwoWeekLow, forwardEps=forwardEps, forwardPE=forwardPE,
        marketCap=marketCap, priceToBook=priceToBook,returnOnAssets=returnOnAssets,
        revenueGrowth=revenueGrowth, targetHighPrice=targetHighPrice, targetLowPrice=targetLowPrice,
        trailingEps=trailingEps, trailingPE=trailingPE, twoHundredDayAverage=twoHundredDayAverage,
        returnOnEquity=returnOnEquity, changefromyesterday=changefromyesterday,
        changefromyesterdayper=changefromyesterdayper,sign=sign,lowPrice=lowPrice,stockDate=stockDate,
        highPrice=highPrice,closePrice=closePrice,volume=volume,Returns=Returns,openPrice=openPrice,
        sma25=sma25,sma50=sma50,possma=possma,sigsma=sigsma,madate=madate,maclose=maclose,
        buyx=buyx,buyy=buyy,sellx=sellx,selly=selly,ema50=ema50,ema25=ema25,emabuyx=emabuyx,emabuyy=emabuyy,
        emasellx=emasellx,emaselly=emaselly,bbmean=bbmean,bbhigh=bbhigh,bblow=bblow,bbbuyx=bbbuyx,
        bbbuyy=bbbuyy,bbsellx=bbsellx,bbselly=bbselly,sig9=sig9,macd=macd,cdbuyx=cdbuyx,cdbuyy=cdbuyy,
        cdsellx=cdsellx,cdselly=cdselly,rsi=rsi,r30=r30,r70=r70,rsbuyx=rsbuyx,rsbuyy=rsbuyy,
        rssellx=rssellx,rsselly=rsselly,showsmabuydate=showsmabuydate,showsmabuyprice=showsmabuyprice,
        showsmaselldate=showsmaselldate,showsmasellprice=showsmasellprice,showemabuydate=showemabuydate,
        showemabuyprice=showemabuyprice,showemaselldate=showemaselldate,showemasellprice=showemasellprice,
        showcdbuydate=showcdbuydate,showcdbuyprice=showcdbuyprice,showcdselldate=showcdselldate,
        showcdsellprice=showcdsellprice,showbbbuydate=showbbbuydate,showbbbuyprice=showbbbuyprice,
        showbbselldate=showbbselldate,showbbsellprice=showbbsellprice,showrsibuydate=showrsibuydate,
        showrsibuyprice=showrsibuyprice,showrsiselldate=showrsiselldate,showrsisellprice=showrsisellprice,
        tom_price=tom_price,changefromtoday=changefromtoday,changefromtodayper=changefromtodayper,predsign=predsign)

    #return render_template(/getdata)  
@app.route("/portfolio", methods=["GET", "POST"])
def portfolio():
    if request.method == "POST":
        ticker = request.form.get("ticker") 
        buyprice=request.form.get("buyprice")
        buydate=request.form.get("buydate")
        numstock=request.form.get("numstock")
if __name__ == "__main__":        # on running python app.py
    app.run(debug=True,host='0.0.0.0')                     # run the flask app 
