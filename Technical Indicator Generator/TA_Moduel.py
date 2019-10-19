#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import talib
import pandas as pd
import os
import talib as ta

""" Function to create the technical indicators """
def get_technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Adj Close'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Adj Close'].rolling(window=21).mean()
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Adj Close'].ewm(com=0.5).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['Adj Close'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Adj Close'].ewm(span=12).mean()
    dataset['MACD'] = (dataset['12ema']-dataset['26ema'])
    
    # Create Momentum
    dataset['momentum'] = dataset['Adj Close']-1
    
    # Create Bollinger Bands
    dataset['20sd'] = dataset['Adj Close'].rolling(20).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create RSI indicator
    dataset['RSI']=ta.RSI(np.array(dataset['Adj Close']))
    
    #Part I: Create Cycle Indicators 
    #Create HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    dataset['HT_DCPERIOD'] = ta.HT_DCPERIOD(np.array(dataset['Adj Close']))
    
    #Create HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    dataset['HT_DCPHASE'] = ta.HT_DCPHASE(np.array(dataset['Adj Close']))
    
    #HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    dataset['HT_TRENDMODE'] = ta.HT_TRENDMODE(np.array(dataset['Adj Close']))
    
    #Part II: Create Volatility Indicators
    #Create Average True Range
    dataset['ATR'] = ta.ATR(np.array(dataset['High']),np.array(dataset['Low']),np.array(dataset['Adj Close']), timeperiod=14)
    
    #Create NATR - Normalized Average True Range
    dataset['NATR'] = ta.NATR(np.array(dataset['High']),np.array(dataset['Low']),np.array(dataset['Adj Close']), timeperiod=14)
    
    #Create TRANGE - True Range
    dataset['TRANGE'] = ta.TRANGE(np.array(dataset['High']),np.array(dataset['Low']),np.array(dataset['Adj Close']))
   
    #Part III Overlap Studies
    #Create DEMA - Double Exponential Moving Average
    dataset['DEMA'] = ta.DEMA(np.array(dataset['Adj Close']), timeperiod=30)
    
                             
    #Create HT_TRENDLINE - Hilbert Transform - Instantaneous Trendline
    dataset['HT_TRENDLINE'] = ta.HT_TRENDLINE(np.array(dataset['Adj Close']))
    
    #Create KAMA - Kaufman Adaptive Moving Average
    dataset['KAMA'] = ta.KAMA(np.array(dataset['Adj Close']), timeperiod=30)

    #Create MIDPOINT - MidPoint over period
    dataset['MIDPOINT'] = ta.MIDPOINT(np.array(dataset['Adj Close']), timeperiod=14)
    
    #Create MIDPRICE - Midpoint Price over period
    dataset['MIDPRICE'] = ta.MIDPRICE(np.array(dataset['High']),np.array(dataset['Low']), timeperiod=14)
    
    #Create SAR - Parabolic SAR
    dataset['SAR']=ta.SAR(np.array(dataset['High']),np.array(dataset['Low']), acceleration=0, maximum=0)
    
    #Create SMA - Simple Moving Average
    dataset['SMA10'] = ta.SMA(np.array(dataset['Adj Close']), timeperiod=10)
    
    #Create T3 - Triple Exponential Moving Average (T3)
    dataset['T3'] = ta.T3(np.array(dataset['Adj Close']), timeperiod=5, vfactor=0)
    
    #Create TRIMA - Triangular Moving Average
    dataset['TRIMA'] = ta.TRIMA(np.array(dataset['Adj Close']), timeperiod=30)
    
    #Create WMA - Weighted Moving Average
    dataset['WMA'] = ta.WMA(np.array(dataset['Adj Close']), timeperiod=30)
    
    #PART IV Momentum Indicators
    #Create ADX - Average Directional Movement Index
    dataset['ADX14'] = ta.ADX(np.array(dataset['High']),np.array(dataset['Low']),np.array(dataset['Adj Close']), timeperiod=14)
    dataset['ADX20'] = ta.ADX(np.array(dataset['High']),np.array(dataset['Low']),np.array(dataset['Adj Close']), timeperiod=20)
    
    #Create ADXR - Average Directional Movement Index Rating
    dataset['ADXR'] = ta.ADXR(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=14)
    
    #Create APO - Absolute Price Oscillator
    dataset['APO'] = ta.APO(np.array(dataset['Adj Close']), fastperiod=12, slowperiod=26, matype=0)

    
    #Create AROONOSC - Aroon Oscillator
    dataset['AROONOSC'] = ta.AROONOSC(np.array(dataset['High']), np.array(dataset['Low']), timeperiod=14)


    #Create BOP - Balance Of Power
    dataset['BOP'] = ta.BOP(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))

    #Create CCI - Commodity Channel Index 
    dataset['CCI3'] = ta.CCI(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=3)
    dataset['CCI5'] = ta.CCI(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=5)
    dataset['CCI10'] = ta.CCI(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=10)
    dataset['CCI14'] = ta.CCI(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=14)
   
    #Create CMO - Chande Momentum Oscillator
    dataset['CMO']= ta.CMO(np.array(dataset['Adj Close']), timeperiod=14)
    
    #Create DX - Directional Movement Index
    dataset['DX']= ta.DX(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=14)

    #Create MINUS_DI - Minus Directional Indicator
    dataset['MINUS_DI'] = ta.MINUS_DI(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=14)


    #Create MINUS_DM - Minus Directional Movement
    dataset['MINUS_DM'] = ta.MINUS_DM(np.array(dataset['High']), np.array(dataset['Low']), timeperiod=14)


    #Create MOM - Momentum 
    dataset['MOM3'] = ta.MOM(np.array(dataset['Adj Close']), timeperiod=3)
    dataset['MOM5'] = ta.MOM(np.array(dataset['Adj Close']), timeperiod=5)
    dataset['MOM10'] = ta.MOM(np.array(dataset['Adj Close']), timeperiod=10)

    #Create PLUS_DI - Plus Directional Indicator
    dataset['PLUS_DI'] = ta.PLUS_DI(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=14)


    #Create PLUS_DM - Plus Directional Movement
    dataset['PLUS_DM'] = ta.PLUS_DM(np.array(dataset['High']), np.array(dataset['Low']), timeperiod=14)

    #Create PPO - Percentage Price Oscillator
    dataset['PPO'] = ta.PPO(np.array(dataset['Adj Close']), fastperiod=12, slowperiod=26, matype=0)

    #Create ROC - Rate of change : ((price/prevPrice)-1)*100
    dataset['ROC'] = ta.ROC(np.array(dataset['Adj Close']), timeperiod=10)

    #Create ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    dataset['ROCP'] = ta.ROCP(np.array(dataset['Adj Close']), timeperiod=10)

    #Create ROCR - Rate of change ratio: (price/prevPrice)
    dataset['ROCR'] = ta.ROCR(np.array(dataset['Adj Close']), timeperiod=10)

    #Create ROCR100 - Rate of change ratio 100 scale: (price/prevPrice)*100
    dataset['ROCR100'] = ta.ROCR100(np.array(dataset['Adj Close']), timeperiod=10)

    #Create RSI - Relative Strength Index
    dataset['RSI5'] = ta.RSI(np.array(dataset['Adj Close']), timeperiod=5)
    dataset['RSI10'] = ta.RSI(np.array(dataset['Adj Close']), timeperiod=10)
    dataset['RSI14'] = ta.RSI(np.array(dataset['Adj Close']), timeperiod=14)
    
    #Create TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    dataset['TRIX'] = ta.TRIX(np.array(dataset['Adj Close']), timeperiod=30)

    #Create ULTOSC - Ultimate Oscillator
    dataset['ULTOSC'] = ta.ULTOSC(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod1=7, timeperiod2=14, timeperiod3=28)

    #Create WILLR - Williams' %R
    dataset['WILLR'] = ta.WILLR(np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), timeperiod=14)
   
    #Part V Pattern Recognition
    #Create  CDL2CROWS - Two Crows
    dataset['CDL2CROWS']  = ta.CDL2CROWS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDL3BLACKCROWS - Three Black Crows
    dataset['CDL3BLACKCROWS']  = ta.CDL3BLACKCROWS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDL3INSIDE - Three Inside Up/Down
    dataset['CDL3INSIDE']  = ta.CDL3INSIDE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDL3LINESTRIKE - Three-Line Strike
    dataset['CDL3LINESTRIKE']  = ta.CDL3LINESTRIKE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDL3OUTSIDE - Three Outside Up/Down
    dataset['CDL3OUTSIDE']  = ta.CDL3OUTSIDE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDL3STARSINSOUTH - Three Stars In The South
    dataset['CDL3STARSINSOUTH ']  = ta.CDL3STARSINSOUTH(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDL3WHITESOLDIERS - Three Advancing White Soldiers
    dataset['CDL3WHITESOLDIERS']  = ta.CDL3WHITESOLDIERS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLABANDONEDBABY - Abandoned Baby
    dataset['CDLABANDONEDBABY']  = ta.CDLABANDONEDBABY(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), penetration=0)
    
    #Create CDLADVANCEBLOCK - Advance Block
    dataset['CDLADVANCEBLOCK']  = ta.CDLADVANCEBLOCK(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLBELTHOLD - Belt-hold
    dataset['CDLBELTHOLD']  = ta.CDLBELTHOLD(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLBREAKAWAY - Breakaway
    dataset['CDLBREAKAWAY']  = ta.CDLBREAKAWAY(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLCLOSINGMARUBOZU - Closing Marubozu
    dataset['CDLCLOSINGMARUBOZU']  = ta.CDLCLOSINGMARUBOZU(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLCONCEALBABYSWALL - Concealing Baby Swalnp.array(dataset['Low'])
    dataset['CDLCONCEALBABYSWALL']  = ta.CDLCONCEALBABYSWALL(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLCOUNTERATTACK - Counterattack
    dataset['CDLCOUNTERATTACK']  = ta.CDLCOUNTERATTACK(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLDARKCLOUDCOVER - Dark Cloud Cover
    dataset['CDLDARKCLOUDCOVER']  =ta. CDLDARKCLOUDCOVER(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), penetration=0)
  
    #Create CDLDOJI - Doji
    dataset['CDLDOJI']  = ta.CDLDOJI(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLDOJISTAR - Doji Star
    dataset['CDLDOJISTAR']  = ta.CDLDOJISTAR(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLDRAGONFLYDOJI - Dragonfly Doji
    dataset['CDLDRAGONFLYDOJI']  = ta.CDLDRAGONFLYDOJI(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLENGULFING - Engulfing Pattern
    dataset['CDLENGULFING']  = ta.CDLENGULFING(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLEVENINGDOJISTAR - Evening Doji Star
    dataset['CDLEVENINGDOJISTAR']  = ta.CDLEVENINGDOJISTAR(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), penetration=0)
   
    #Create CDLEVENINGSTAR - Evening Star
    dataset['CDLEVENINGSTAR']  = ta.CDLEVENINGSTAR(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), penetration=0)
   
    #Create CDLGAPSIDESIDEWHITE - Up/Down-gap side-by-side white lines
    dataset['CDLGAPSIDESIDEWHITE']  = ta.CDLGAPSIDESIDEWHITE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLGRAVESTONEDOJI - Gravestone Doji
    dataset['CDLGRAVESTONEDOJI']  = ta.CDLGRAVESTONEDOJI(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLHAMMER - Hammer
    dataset['CDLHAMMER']  = ta.CDLHAMMER(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLHANGINGMAN - Hanging Man
    dataset['CDLHANGINGMAN']  = ta.CDLHANGINGMAN(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLHARAMI - Harami Pattern
    dataset['CDLHARAMI']  = ta.CDLHARAMI(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLHARAMICROSS - Harami Cross Pattern
    dataset['CDLHARAMICROSS']  = ta.CDLHARAMICROSS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLHIGHWAVE - High-Wave Candle
    dataset['CDLHIGHWAVE']  =  ta.CDLHIGHWAVE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLHIKKAKE - Hikkake Pattern
    dataset['CDLHIKKAKE']  = ta.CDLHIKKAKE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLHIKKAKEMOD - Modified Hikkake Pattern
    dataset['CDLHIKKAKEMOD']  = ta.CDLHIKKAKEMOD(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLHOMINGPIGEON - Homing Pigeon
    dataset['CDLHOMINGPIGEON']  = ta.CDLHOMINGPIGEON(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLIDENTICAL3CROWS - Identical Three Crows
    dataset['CDLIDENTICAL3CROWS']  = ta.CDLIDENTICAL3CROWS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLINNECK - In-Neck Pattern
    dataset['CDLINNECK']  = ta.CDLINNECK(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLINVERTEDHAMMER - Inverted Hammer
    dataset['CDLINVERTEDHAMMER']  = ta.CDLINVERTEDHAMMER(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLKICKING - Kicking
    dataset['CDLKICKING']  = ta.CDLKICKING(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLKICKINGBYLENGTH - Kicking - bull/bear determined by the longer marubozu
    dataset['CDLKICKINGBYLENGTH']  = ta.CDLKICKINGBYLENGTH(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLLADDERBOTTOM - Ladder Bottom
    dataset['CDLLADDERBOTTOM']  = ta.CDLLADDERBOTTOM(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
 
    #Create CDLLONGLEGGEDDOJI - Long Legged Doji
    dataset['CDLLONGLEGGEDDOJI']  = ta.CDLLONGLEGGEDDOJI(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
  
    #Create CDLLONGLINE - Long Line Candle
    dataset['CDLLONGLINE']  = ta.CDLLONGLINE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLMARUBOZU - Marubozu
    dataset['CDLMARUBOZU']  = ta.CDLMARUBOZU(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLMATCHINGLOW - Matching Low
    dataset['CDLMATCHINGLOW']  = ta.CDLMATCHINGLOW(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    #Create CDLMATHOLD - Mat Hold
    dataset['CDLMATHOLD']  = ta.CDLMATHOLD(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), penetration=0)
        
    #Create CDLMORNINGDOJISTAR - Morning Doji Star
    dataset['CDLMORNINGDOJISTAR']  = ta.CDLMORNINGDOJISTAR(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), penetration=0)
    
    #Create CDLMORNINGSTAR - Morning Star
    dataset['CDLMORNINGSTAR']  = ta.CDLMORNINGSTAR(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']), penetration=0)
    
    #Create CDLONNECK - On-Neck Pattern
    dataset['CDLONNECK']  = ta.CDLONNECK(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLPIERCING - Piercing Pattern
    dataset['CDLPIERCING']  = ta.CDLPIERCING(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLRICKSHAWMAN - Rickshaw Man
    dataset['CDLRICKSHAWMAN']  = ta.CDLRICKSHAWMAN(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLRISEFALL3METHODS - Rising/Falling Three Methods
    dataset['CDLRISEFALL3METHODS']  = ta.CDLRISEFALL3METHODS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLSEPARATINGLINES - Separating Lines
    dataset['CDLSEPARATINGLINES']  = ta.CDLSEPARATINGLINES(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLSHOOTINGSTAR - Shooting Star
    dataset['CDLSHOOTINGSTAR']  = ta.CDLSHOOTINGSTAR(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLSHORTLINE - Short Line Candle
    dataset['CDLSHORTLINE']  = ta.CDLSHORTLINE(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLSPINNINGTOP - Spinning Top
    dataset['CDLSPINNINGTOP']  = ta.CDLSPINNINGTOP(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLSTALLEDPATTERN - Stalled Pattern
    dataset['CDLSTALLEDPATTERN']  = ta.CDLSTALLEDPATTERN(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLSTICKSANDWICH - Stick Sandwich
    dataset['CDLSTICKSANDWICH']  = ta.CDLSTICKSANDWICH(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLTAKURI - Takuri (Dragonfly Doji with very long np.array(dataset['Low'])er shadow)
    dataset['CDLTAKURI']  = ta.CDLTAKURI(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLTASUKIGAP - Tasuki Gap
    dataset['CDLTASUKIGAP']  = ta.CDLTASUKIGAP(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLTHRUSTING - Thrusting Pattern
    dataset['CDLTHRUSTING']  = ta.CDLTHRUSTING(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLTRISTAR - Tristar Pattern
    dataset['CDLTRISTAR']  = ta.CDLTRISTAR(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLUNIQUE3RIVER - Unique 3 River
    dataset['CDLUNIQUE3RIVER']  = ta.CDLUNIQUE3RIVER(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLUPSIDEGAP2CROWS - Upside Gap Two Crows
    dataset['CDLUPSIDEGAP2CROWS']  = ta.CDLUPSIDEGAP2CROWS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
    
    #Create CDLXSIDEGAP3METHODS - Upside/Downside Gap Three Methods
    dataset['CDLXSIDEGAP3METHODS']  = ta.CDLXSIDEGAP3METHODS(np.array(dataset['Open']), np.array(dataset['High']), np.array(dataset['Low']), np.array(dataset['Adj Close']))
   
    return dataset

    

