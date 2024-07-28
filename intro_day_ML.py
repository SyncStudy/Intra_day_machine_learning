# from AlgorithmImports import *

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import RobustScaler
# from sklearn.ensemble import RandomForestClassifier
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense

# class MLIntradayTrading(QCAlgorithm):
#     def Initialize(self):
#         self.SetStartDate(2024, 1, 1)    # Set Start Date
#         self.SetEndDate(2024, 6, 1)      # Set End Date
#         self.SetCash(100000)             # Set Strategy Cash
        
#         # Add S&P 500 stocks to the universe
#         self.symbols = [self.AddEquity(ticker, Resolution.Minute).Symbol for ticker in self.SPYComponents()]
        
#         # Rolling window to hold feature data
#         self.lookback = 240
#         self.window = {symbol: RollingWindow[TradeBar](self.lookback) for symbol in self.symbols}
        
#         # Parameters
#         self.features = {}
#         self.scaler = RobustScaler()
        
#         # Schedule training and trading
#         self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(15, 30), self.TrainAndTrade)
    
#     # def SPYComponents(self):
#     #     # This function should return a list of S&P 500 tickers
#     #     return ["AAPL", "MSFT", "GOOG", "AMZN", "FB", "BRK.B", "JNJ", "JPM", "V", "PG"]  # Add more as needed
#     def SPYComponents(self):
#     # Full list of S&P 500 tickers as of a specific date
#         return [
#             "AAPL", "MSFT", "GOOGL", "AMZN", "FB", "BRK.B", "JNJ", "JPM", "V", "PG", "UNH", "NVDA", "HD",
#             "PYPL", "MA", "DIS", "ADBE", "NFLX", "INTC", "CMCSA", "VZ", "PFE", "KO", "PEP", "MRK", "T",
#             "CSCO", "ABT", "CRM", "ACN", "ABBV", "XOM", "CVX", "MDT", "NKE", "LLY", "WFC", "COST", "AVGO",
#             "QCOM", "NEE", "TXN", "LIN", "PM", "HON", "ORCL", "AMGN", "UPS", "SBUX", "MS", "MMM", "UNP",
#             "RTX", "LOW", "IBM", "GS", "CAT", "CVS", "ISRG", "BLK", "LMT", "GE", "PLD", "BA", "TMO", "SYK",
#             "BDX", "SCHW", "MDLZ", "AMT", "DUK", "CI", "SO", "AXP", "ADP", "ADI", "SPGI", "DHR", "BMY",
#             "NOW", "MO", "ZTS", "FIS", "GILD", "MMM", "C", "ETN", "CL", "DE", "AMAT", "MMM", "TGT", "PNC",
#             "MU", "EL", "BKNG", "APD", "INTU", "MMC", "TFC", "HUM", "BSX", "ADSK", "GM", "EMR", "ADP",
#             "DG", "REGN", "VRTX", "MSI", "AON", "WM", "AEP", "FDX", "TJX", "EW", "NOC", "ICE", "ADBE",
#             "D", "MET", "SPG", "SRE", "MRNA", "FISV", "AIG", "NEM", "IQV", "PSA", "ILMN", "COF", "HCA",
#             "IDXX", "APTV", "WBA", "LRCX", "A", "STZ", "BIIB", "TRV", "SNPS", "ZBH", "DLTR", "WLTW",
#             "GPN", "NDAQ", "PPG", "CSX", "TT", "JCI", "EBAY", "MNST", "FTV", "HAL", "HIG", "CTSH", "OTIS",
#             "SYY", "ORLY", "ATVI", "NUE", "SWK", "DHI", "PGR", "ED", "WDC", "KHC", "ALL", "EXC", "MAR",
#             "STT", "HSY", "APH", "ETSY", "PEG", "BAX", "CBRE", "EFX", "CTAS", "CDNS", "VRSK", "CERN",
#             "MCK", "BKR", "FLT", "WEC", "WMB", "ETR", "CMS", "MSCI", "DXCM", "ROST", "KMB", "KEYS",
#             "MCHP", "FAST", "CINF", "HPE", "DTE", "CPRT", "KMX", "MTD", "PPL", "ROK", "MPC", "AWK",
#             "EXPE", "RMD", "GRMN", "WAT", "ES", "AEE", "MKC", "DOV", "TDY", "LNT", "PAYX", "TER", "CAG",
#             "ESS", "IFF", "FE", "RE", "WELL", "AVB", "FMC", "STE", "XYL", "VTR", "ATO", "AKAM", "CRL",
#             "HWM", "HAS", "FRT", "IRM", "HST", "SEE", "VMC", "NLSN", "WRB", "CNP", "CMS", "EXPD", "HOLX",
#             "ABC", "K", "TSN", "SJM", "NTRS", "TYL", "JKHY", "BXP", "DRE", "FFIV", "IVZ", "ALGN", "PFG",
#             "HRL", "UDR", "PKI", "XYL", "MAS", "HES", "WHR", "MKTX", "EQT", "WY", "CHRW", "WRB", "NWSA",
#             "NWS", "ZBRA", "L", "CZR", "OGN", "MOS", "NLOK", "AOS", "AES", "NRG", "CMA", "ETR", "CLX",
#             "XEL", "DRI", "DVA", "AES", "NI", "CMS", "O", "LUV", "QRVO", "J", "GL", "ZBH", "PKI", "CDW",
#             "BEN", "DISH", "PWR", "ABMD", "AAL", "MRO", "DHR", "TDG", "CHD", "VFC", "VTRS", "TAP", "FOXA",
#             "FOXA", "CCL", "RL", "NWL", "MHK", "UAL", "WYNN", "RCL", "PENN", "FHN", "APA", "ALB", "TPR",
#             "HBI", "HAS", "ALLE", "FLS", "SEE", "DOV", "L", "RE", "UHS", "TROW", "VNO", "NUE", "PKG",
#             "AKAM", "SBAC", "GRMN", "IRM", "KIM", "KEYS", "SIVB", "LYV", "CE", "DGX", "ZTS", "CME",
#             "RSG", "STLD", "VAR", "BXP", "AVTR", "CZR", "MTB", "ARE", "DRE", "TRMB", "GL", "VRSN", "CDNS",
#             "EQR", "NDAQ", "SBAC", "PFG", "CPT", "MAA", "HST", "SYY", "WY", "STZ", "CMA", "FFIV", "JKHY",
#             "SIVB", "AES", "RCL", "LYV", "CNP", "DTE", "CDW", "AES", "RSG", "MOS", "HES", "SEE", "SYY",
#             "AES", "STT", "HRL", "TROW", "FFIV", "L", "SBAC", "CHRW", "WRB", "SBAC", "AEE", "SBAC", "MOS",
#             "SYY", "SBAC", "AES", "MOS", "SBAC", "RCL", "AES", "SYY", "AES", "AES", "AES", "AES", "AES"
#         ]
    
#     def OnData(self, data):
#         # Update rolling window
#         for symbol in self.symbols:
#             if symbol in data and data[symbol] is not None:
#                 self.window[symbol].Add(data[symbol])
#                 # self.Debug(f"Added data for {symbol}: {data[symbol].Close}")

#     def TrainAndTrade(self):
#         # self.Debug("TrainAndTrade triggered")
#         # Prepare training data
#         feature_data = []
#         labels = []
        
#         for symbol in self.symbols:
#             if self.window[symbol].IsReady:
#                 closes = [bar.Close for bar in self.window[symbol] if bar is not None]
#                 opens = [bar.Open for bar in self.window[symbol] if bar is not None]
#                 if len(closes) >= self.lookback and len(opens) >= self.lookback:
#                     df = pd.DataFrame({'Close': closes, 'Open': opens})
#                     df['r_intra'] = df['Close'].pct_change().shift(-1)
#                     df['r_close'] = df['Close'].pct_change()
#                     df['r_open'] = (df['Close'] - df['Open']) / df['Open']
                    
#                     # self.Debug(f"Before dropna - Symbol: {symbol}, df length: {len(df)}, df: {df.head()}")
#                     df.dropna(inplace=True)  # Drop rows with NaNs
#                     # self.Debug(f"After dropna - Symbol: {symbol}, df length: {len(df)}, df: {df.head()}")

#                     if len(df) >= self.lookback * 0.6:  # Allow for some missing data
#                         features = df[['r_intra', 'r_close', 'r_open']].values
#                         # self.Debug(f"Appending features for {symbol}: {features}")
#                         feature_data.append(features)
#                         labels.append(1 if df['r_intra'].iloc[-1] > 0 else 0)
        
#         self.Debug(f"Feature data length: {len(feature_data)}")
#         if len(feature_data) < 10:
#             self.Debug("Not enough data to train models")
#             return  # Ensure enough data to train models
        
#         # Convert feature_data to a readable format for debugging
#         feature_data_readable = [f.tolist() for f in feature_data]
#         # self.Debug(f"Feature data: {feature_data_readable}")
        
#         X = np.array(feature_data)
#         y = np.array(labels)
        
#         # self.Debug(f"X shape before scaling: {X.shape}")
        
#         try:
#             X_reshaped = X.reshape(-1, 3)  # Reshape to 2D array with 3 features per row
#             X_scaled = self.scaler.fit_transform(X_reshaped).reshape(X.shape)
#             self.Debug(f"X shape after scaling: {X_scaled.shape}")
#         except ValueError as e:
#             self.Debug(f"Reshape error: {e}")
#             return
        
#         # Train LSTM
#         lstm_model = Sequential()
#         lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
#         lstm_model.add(LSTM(50))
#         lstm_model.add(Dense(1, activation='sigmoid'))
#         lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#         lstm_model.fit(X_scaled, y, epochs=10, batch_size=64, verbose=0)
        
#         # Train Random Forest
#         rf_model = RandomForestClassifier(n_estimators=100)
#         rf_model.fit(X_scaled.reshape(X_scaled.shape[0], -1), y)
        
#         # Generate predictions
#         predictions = []
#         for symbol in self.symbols:
#             if self.window[symbol].IsReady:
#                 closes = [bar.Close for bar in self.window[symbol] if bar is not None]
#                 opens = [bar.Open for bar in self.window[symbol] if bar is not None]
#                 if len(closes) >= self.lookback and len(opens) >= self.lookback:
#                     df = pd.DataFrame({'Close': closes, 'Open': opens})
#                     df['r_intra'] = df['Close'].pct_change().shift(-1)
#                     df['r_close'] = df['Close'].pct_change()
#                     df['r_open'] = (df['Close'] - df['Open']) / df['Open']

#                     df.dropna(inplace=True)  # Drop rows with NaNs

#                     if len(df) >= self.lookback * 0.8:  # Allow for some missing data
#                         features = df[['r_intra', 'r_close', 'r_open']].values
#                         self.Debug(f"Transforming features for {symbol}: {features}")
#                         X_new = self.scaler.transform(features).reshape(1, features.shape[0], features.shape[1])
#                         self.Debug(f"X_new shape after transforming: {X_new.shape}")
#                         lstm_pred = lstm_model.predict(X_new)
#                         rf_pred = rf_model.predict(X_new.reshape(1, -1))
#                         predictions.append((symbol, lstm_pred[0][0], rf_pred[0]))
        
#         self.Debug(f"Number of predictions: {len(predictions)}")
#         if not predictions:
#             self.Debug("No predictions made")
#             return
        
#         # Long top 10, short bottom 10 based on LSTM predictions
#         predictions.sort(key=lambda x: x[1], reverse=True)
#         long_symbols = [x[0] for x in predictions[:10]]
#         short_symbols = [x[0] for x in predictions[-10:]]
        
#         # Liquidate all holdings
#         for symbol in self.Portfolio.Keys:
#             if symbol not in long_symbols and symbol not in short_symbols:
#                 self.Liquidate(symbol)
        
#         # Set holdings
#         for symbol in long_symbols:
#             self.SetHoldings(symbol, 0.1)  # Long top 10
        
#         for symbol in short_symbols:
#             self.SetHoldings(symbol, -0.1)  # Short bottom 10


## 

###########################################################################
# Increased Lookback Period: The lookback period has been increased to 480 (8 hours) to capture more data and potentially more robust patterns.
# Additional Features: Added new features like high_low_range and close_open_range to capture more information about price movements.
# Model Hyperparameters: Increased the epochs for the LSTM model to 20 and adjusted the RandomForestClassifier with n_estimators=200 and max_depth=10.
# Data Normalization: Ensured consistent data normalization with the correct number of features.
# Debugging: Additional debug statements to monitor data transformations and ensure correctness.



###########################################################################

from AlgorithmImports import *

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class MLIntradayTrading(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2024, 6, 10)    # Set Start Date
        self.SetEndDate(2024, 6, 12)      # Set End Date
        self.SetCash(100000)             # Set Strategy Cash
        
        # Add S&P 500 stocks to the universe
        self.symbols = [self.AddEquity(ticker, Resolution.Minute).Symbol for ticker in self.SPYComponents()]
        
        # Rolling window to hold feature data
        self.lookback = 480  # Increased lookback period
        self.window = {symbol: RollingWindow[TradeBar](self.lookback) for symbol in self.symbols}
        
        # Parameters
        self.features = {}
        self.scaler = RobustScaler()
        
        # Schedule training and trading
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.At(15, 30), self.TrainAndTrade)
    
    def SPYComponents(self):
        # Full list of S&P 500 tickers as of a specific date
        return [
            "AAPL", "MSFT", "GOOGL", "AMZN", "FB", "BRK.B", "JNJ", "JPM", "V", "PG", "UNH", "NVDA", "HD",
            "PYPL", "MA", "DIS", "ADBE", "NFLX", "INTC", "CMCSA", "VZ", "PFE", "KO", "PEP", "MRK", "T",
            "CSCO", "ABT", "CRM", "ACN", "ABBV", "XOM", "CVX", "MDT", "NKE", "LLY", "WFC", "COST", "AVGO",
            "QCOM", "NEE", "TXN", "LIN", "PM", "HON", "ORCL", "AMGN", "UPS", "SBUX", "MS", "MMM", "UNP",
            "RTX", "LOW", "IBM", "GS", "CAT", "CVS", "ISRG", "BLK", "LMT", "GE", "PLD", "BA", "TMO", "SYK",
            "BDX", "SCHW", "MDLZ", "AMT", "DUK", "CI", "SO", "AXP", "ADP", "ADI", "SPGI", "DHR", "BMY",
            "NOW", "MO", "ZTS", "FIS", "GILD", "MMM", "C", "ETN", "CL", "DE", "AMAT", "MMM", "TGT", "PNC",
            "MU", "EL", "BKNG", "APD", "INTU", "MMC", "TFC", "HUM", "BSX", "ADSK", "GM", "EMR", "ADP",
            "DG", "REGN", "VRTX", "MSI", "AON", "WM", "AEP", "FDX", "TJX", "EW", "NOC", "ICE", "ADBE",
            "D", "MET", "SPG", "SRE", "MRNA", "FISV", "AIG", "NEM", "IQV", "PSA", "ILMN", "COF", "HCA",
            "IDXX", "APTV", "WBA", "LRCX", "A", "STZ", "BIIB", "TRV", "SNPS", "ZBH", "DLTR", "WLTW",
            "GPN", "NDAQ", "PPG", "CSX", "TT", "JCI", "EBAY", "MNST", "FTV", "HAL", "HIG", "CTSH", "OTIS",
            "SYY", "ORLY", "ATVI", "NUE", "SWK", "DHI", "PGR", "ED", "WDC", "KHC", "ALL", "EXC", "MAR",
            "STT", "HSY", "APH", "ETSY", "PEG", "BAX", "CBRE", "EFX", "CTAS", "CDNS", "VRSK", "CERN",
            "MCK", "BKR", "FLT", "WEC", "WMB", "ETR", "CMS", "MSCI", "DXCM", "ROST", "KMB", "KEYS",
            "MCHP", "FAST", "CINF", "HPE", "DTE", "CPRT", "KMX", "MTD", "PPL", "ROK", "MPC", "AWK",
            "EXPE", "RMD", "GRMN", "WAT", "ES", "AEE", "MKC", "DOV", "TDY", "LNT", "PAYX", "TER", "CAG",
            "ESS", "IFF", "FE", "RE", "WELL", "AVB", "FMC", "STE", "XYL", "VTR", "ATO", "AKAM", "CRL",
            "HWM", "HAS", "FRT", "IRM", "HST", "SEE", "VMC", "NLSN", "WRB", "CNP", "CMS", "EXPD", "HOLX",
            "ABC", "K", "TSN", "SJM", "NTRS", "TYL", "JKHY", "BXP", "DRE", "FFIV", "IVZ", "ALGN", "PFG",
            "HRL", "UDR", "PKI", "XYL", "MAS", "HES", "WHR", "MKTX", "EQT", "WY", "CHRW", "WRB", "NWSA",
            "NWS", "ZBRA", "L", "CZR", "OGN", "MOS", "NLOK", "AOS", "AES", "NRG", "CMA", "ETR", "CLX",
            "XEL", "DRI", "DVA", "AES", "NI", "CMS", "O", "LUV", "QRVO", "J", "GL", "ZBH", "PKI", "CDW",
            "BEN", "DISH", "PWR", "ABMD", "AAL", "MRO", "DHR", "TDG", "CHD", "VFC", "VTRS", "TAP", "FOXA",
            "FOXA", "CCL", "RL", "NWL", "MHK", "UAL", "WYNN", "RCL", "PENN", "FHN", "APA", "ALB", "TPR",
            "HBI", "HAS", "ALLE", "FLS", "SEE", "DOV", "L", "RE", "UHS", "TROW", "VNO", "NUE", "PKG",
            "AKAM", "SBAC", "GRMN", "IRM", "KIM", "KEYS", "SIVB", "LYV", "CE", "DGX", "ZTS", "CME",
            "RSG", "STLD", "VAR", "BXP", "AVTR", "CZR", "MTB", "ARE", "DRE", "TRMB", "GL", "VRSN", "CDNS",
            "EQR", "NDAQ", "SBAC", "PFG", "CPT", "MAA", "HST", "SYY", "WY", "STZ", "CMA", "FFIV", "JKHY",
            "SIVB", "AES", "RCL", "LYV", "CNP", "DTE", "CDW", "AES", "RSG", "MOS", "HES", "SEE", "SYY",
            "AES", "STT", "HRL", "TROW", "FFIV", "L", "SBAC", "CHRW", "WRB", "SBAC", "AEE", "SBAC", "MOS",
            "SYY", "SBAC", "AES", "MOS", "SBAC", "RCL", "AES", "SYY", "AES", "AES", "AES", "AES", "AES"
        ]
    
    def OnData(self, data):
        # Update rolling window
        for symbol in self.symbols:
            if symbol in data and data[symbol] is not None:
                self.window[symbol].Add(data[symbol])
                # self.Debug(f"Added data for {symbol}: {data[symbol].Close}")

    def TrainAndTrade(self):
        # self.Debug("TrainAndTrade triggered")
        # Prepare training data
        feature_data = []
        labels = []
        
        for symbol in self.symbols:
            if self.window[symbol].IsReady:
                closes = [bar.Close for bar in self.window[symbol] if bar is not None]
                opens = [bar.Open for bar in self.window[symbol] if bar is not None]
                highs = [bar.High for bar in self.window[symbol] if bar is not None]
                lows = [bar.Low for bar in self.window[symbol] if bar is not None]
                
                if len(closes) >= self.lookback and len(opens) >= self.lookback:
                    df = pd.DataFrame({'Close': closes, 'Open': opens, 'High': highs, 'Low': lows})
                    df['r_intra'] = df['Close'].pct_change().shift(-1)
                    df['r_close'] = df['Close'].pct_change()
                    df['r_open'] = (df['Close'] - df['Open']) / df['Open']
                    df['high_low_range'] = (df['High'] - df['Low']) / df['Low']
                    df['close_open_range'] = (df['Close'] - df['Open']) / df['Open']

                    df.dropna(inplace=True)  # Drop rows with NaNs

                    if len(df) >= self.lookback * 0.8:  # Allow for some missing data
                        features = df[['r_intra', 'r_close', 'r_open', 'high_low_range', 'close_open_range']].values
                        feature_data.append(features)
                        labels.append(1 if df['r_intra'].iloc[-1] > 0 else 0)
        
        self.Debug(f"Feature data length: {len(feature_data)}")
        if len(feature_data) < 10:
            self.Debug("Not enough data to train models")
            return  # Ensure enough data to train models
        
        # Convert feature_data to a readable format for debugging
        feature_data_readable = [f.tolist() for f in feature_data]
        # self.Debug(f"Feature data: {feature_data_readable}")
        
        X = np.array(feature_data)
        y = np.array(labels)
        
        try:
            X_reshaped = X.reshape(-1, 5)  # Reshape to 2D array with 5 features per row
            X_scaled = self.scaler.fit_transform(X_reshaped).reshape(X.shape)
            self.Debug(f"X shape after scaling: {X_scaled.shape}")
        except ValueError as e:
            self.Debug(f"Reshape error: {e}")
            return
        
        # Train LSTM
        lstm_model = Sequential()
        lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_scaled.shape[1], X_scaled.shape[2])))
        lstm_model.add(LSTM(50))
        lstm_model.add(Dense(1, activation='sigmoid'))
        lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        lstm_model.fit(X_scaled, y, epochs=20, batch_size=64, verbose=0)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=200, max_depth=10)
        rf_model.fit(X_scaled.reshape(X_scaled.shape[0], -1), y)
        
        # Generate predictions
        predictions = []
        for symbol in self.symbols:
            if self.window[symbol].IsReady:
                closes = [bar.Close for bar in self.window[symbol] if bar is not None]
                opens = [bar.Open for bar in self.window[symbol] if bar is not None]
                highs = [bar.High for bar in self.window[symbol] if bar is not None]
                lows = [bar.Low for bar in self.window[symbol] if bar is not None]
                
                if len(closes) >= self.lookback and len(opens) >= self.lookback:
                    df = pd.DataFrame({'Close': closes, 'Open': opens, 'High': highs, 'Low': lows})
                    df['r_intra'] = df['Close'].pct_change().shift(-1)
                    df['r_close'] = df['Close'].pct_change()
                    df['r_open'] = (df['Close'] - df['Open']) / df['Open']
                    df['high_low_range'] = (df['High'] - df['Low']) / df['Low']
                    df['close_open_range'] = (df['Close'] - df['Open']) / df['Open']

                    df.dropna(inplace=True)  # Drop rows with NaNs

                    if len(df) >= self.lookback * 0.8:  # Allow for some missing data
                        features = df[['r_intra', 'r_close', 'r_open', 'high_low_range', 'close_open_range']].values
                        self.Debug(f"Transforming features for {symbol}: {features}")
                        X_new = self.scaler.transform(features).reshape(1, features.shape[0], features.shape[1])
                        self.Debug(f"X_new shape after transforming: {X_new.shape}")
                        lstm_pred = lstm_model.predict(X_new)
                        rf_pred = rf_model.predict(X_new.reshape(1, -1))

                        # Only make a prediction if confidence is high
                        if lstm_pred[0][0] > 0.7 or rf_pred[0] > 0.7:
                            predictions.append((symbol, lstm_pred[0][0], rf_pred[0]))
                        # predictions.append((symbol, lstm_pred[0][0], rf_pred[0]))
        
        self.Debug(f"Number of predictions: {len(predictions)}")
        if not predictions:
            self.Debug("No predictions made")
            return
        
        # Long top 10, short bottom 10 based on LSTM predictions
        predictions.sort(key=lambda x: x[1], reverse=True)
        long_symbols = [x[0] for x in predictions[:10]]
        short_symbols = [x[0] for x in predictions[-10:]]
        
        # Liquidate all holdings
        for symbol in self.Portfolio.Keys:
            if symbol not in long_symbols and symbol not in short_symbols:
                self.Liquidate(symbol)
        
        # Set holdings
        for symbol in long_symbols:
            self.SetHoldings(symbol, 0.1)  # Long top 10
        
        for symbol in short_symbols:
            self.SetHoldings(symbol, -0.1)  # Short bottom 10



