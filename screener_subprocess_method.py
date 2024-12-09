import pandas as pd
import talib
import numpy as np
import yfinance as yf
import json
import csv
from PyQt6.QtCore import QThread, pyqtSignal, QDate, Qt
from PyQt6.QtWidgets import QScrollArea, QSplitter, QSlider, QDateEdit, QComboBox, QMessageBox, QTableWidgetItem, QTableWidget, QPushButton, QMainWindow, QLabel, QVBoxLayout, QProgressBar, QWidget, QApplication, QLineEdit, QCheckBox
import matplotlib.pyplot as plt
import main
import subprocess

class NumericTableWidgetItem(QTableWidgetItem):
    def __lt__(self, other):
        return float(self.text()) < float(other.text())

class LoadingWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Loading Stock Data...")
        self.setGeometry(100, 100, 300, 100)
        
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.progress = QProgressBar(self)
        self.progress.setGeometry(0, 0, 300, 25)
        self.progress.setMaximum(100)
        self.layout.addWidget(self.progress)

    def update_progress(self, progress):
        self.progress.setValue(progress)

# Properties of Stocks
class Stock:
    def __init__(self, symbol, sector, market_cap, dividend_yield, pe_ratio, performance_52_week, obv, adx, average_volume_10day,macd,macd_signal,macd_buy, new_high, number_of_new_highs, macd_hist_data, total_score, result):
        self.symbol = symbol
        self.sector = sector
        self.market_cap = market_cap
        self.dividend_yield = dividend_yield
        self.pe_ratio = pe_ratio
        self.performance_52_week = performance_52_week
        self.obv = obv[-1] if len(obv) else None
        self.adx = adx[-1] if len(adx) else None
        self.average_volume_10day = average_volume_10day
        self.macd = macd[-1] if len(macd) > 0 else None
        self.macd_signal = macd_signal[-1] if len(macd_signal) > 0 else None
        self.macd_buy = macd_buy
        self.new_high = new_high
        self.number_of_new_highs = number_of_new_highs
        self.macd_hist_data = macd_hist_data
        self.weighted_score = total_score
        self.advisor_result = result

# Screener class does filtering
class StockScreener:
    def __init__(self, stocks):
        self.stocks = stocks

    # This happens whenever the filter button is clicked
    def filter_stocks(self, min_market_cap=None, min_dividend_yield=None, max_pe_ratio=None, min_performance_52_week=None, min_obv=None, min_adx=None, sector_filter= None, volume_filter = None, macd_filter = None, new_highs = False, number_of_new_highs = None):
        filtered_stocks = []
        for stock in self.stocks:
            if min_market_cap is not None and stock.market_cap < min_market_cap:
                continue
            if min_dividend_yield is not None and stock.dividend_yield < min_dividend_yield:
                continue
            if max_pe_ratio is not None and float(stock.pe_ratio) > max_pe_ratio:
                continue
            if min_performance_52_week is not None and float(stock.performance_52_week) < min_performance_52_week:
                continue
            if min_obv is not None and stock.obv < min_obv:
                continue
            if min_adx is not None and stock.adx < min_adx:
                continue
            if sector_filter is not None and stock.sector != sector_filter:
                continue
            if volume_filter is not None and float(stock.average_volume_10day) < volume_filter:
                continue
            if macd_filter is not None and stock.macd < macd_filter:
                continue
            if new_highs and not stock.new_high:
                continue
            if number_of_new_highs and stock.number_of_new_highs < number_of_new_highs:
                continue

            filtered_stocks.append(stock)

        return filtered_stocks

    def to_dataframe(self, stocks=None):
        if stocks is None:
            stocks = self.stocks

        data = {
            'Symbol': [stock.symbol for stock in stocks],
            'Sector': [stock.sector for stock in stocks],
            'Market Cap': [stock.market_cap for stock in stocks],
            'Dividend Yield': [stock.dividend_yield for stock in stocks],
            'P/E Ratio': [stock.pe_ratio for stock in stocks],
            '52-Week Performance': [stock.performance_52_week for stock in stocks],
            'obv': [stock.obv for stock in stocks],
            'adx': [stock.adx for stock in stocks],
            'average_volume_10day': [stock.average_volume_10day for stock in stocks]
        }

        return pd.DataFrame(data)

# App
class StockScreenerApp(QMainWindow):
    def __init__(self, ticker_list, stock_screener=None):
        super().__init__()
        self.ticker_list = ticker_list
        self.filtered_stocks_list = []
        self.top_scores = []
        self.stock_screener = stock_screener

        #UI init
        self.setWindowTitle("Stock Screener")
        self.setGeometry(100, 100, 1000, 600)

        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.setCentralWidget(self.splitter)

        self.scrollArea = QScrollArea()
        self.splitter.addWidget(self.scrollArea)
        self.scrollArea.setWidgetResizable(True)

        self.central_widget = QWidget()
        self.scrollArea.setWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.date_input_window = DateRange()

        self.sector_filter = QLineEdit()
        self.min_market_cap = QLineEdit()
        self.min_market_cap_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_dividend_yield = QLineEdit()
        self.min_dividend_yield_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_pe_ratio = QLineEdit()
        self.max_pe_ratio_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_performance_52_week = QLineEdit()
        self.min_performance_52_week_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_filter = QLineEdit()
        self.volume_filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_obv = QLineEdit()
        self.min_obv_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_adx = QLineEdit()
        self.min_adx_slider = QSlider(Qt.Orientation.Horizontal)
        self.macd_filter = QLineEdit()
        self.macd_filter_slider = QSlider(Qt.Orientation.Horizontal)
        self.number_of_new_highs = QLineEdit() 
        self.number_of_new_highs_slider = QSlider(Qt.Orientation.Horizontal)

        #Loading window
        self.loading_window = LoadingWindow()

        if stock_screener is not None:
            self.create_widgets()
        self.hide()

    def add_numeric_filter(self, label_text, min_value, max_value, scaling_factor, line_edit, slider):
        label = QLabel(label_text)
        self.layout.addWidget(label)
        
        slider.setMinimum(int(min_value * scaling_factor))
        try:
            slider.setMaximum(int(max_value * scaling_factor))
        except OverflowError:
            if max_value == float('infinity'):
                slider.setMaximum(int(2500000))
            else:
                raise
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        try:
            slider.setTickInterval(int((max_value - min_value) / 10 * scaling_factor))
        except OverflowError:
            if max_value == float('infinity'):
                slider.setTickInterval(int(2500000))
            else:
                raise

        line_edit.textChanged.connect(lambda text: self.update_slider(text, slider, scaling_factor))
        slider.valueChanged.connect(lambda value: self.update_line_edit(value, line_edit, scaling_factor))

        self.layout.addWidget(line_edit)
        self.layout.addWidget(slider)

    def update_line_edit(self, value, line_edit, scaling_factor):
        line_edit.setText(str(value / scaling_factor))

    def update_slider(self, text, slider, scaling_factor):
        try:
            value = int(float(text) * scaling_factor)
        except ValueError:
            value = 0
        slider.setValue(value)

    def retrieve_date(self):
        self.date_input_window = DateRange()
        self.date_input_window.datesSelected.connect(self.set_dates)
        self.date_input_window.closeEvent = self.on_date_window_closed
        self.date_input_window.show()

    def set_dates(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

    def on_date_window_closed(self, event):
        self.start_loading_thread()

    def loading_done(self):
        self.loading_window.close()
        self.stock_screener = StockScreener(self.thread.stocks)
        
        #slider perameters
        self.slider_max_market_cap = max([stock.market_cap for stock in self.thread.stocks])
        self.slider_div_yield = max([float(stock.dividend_yield) for stock in self.thread.stocks]) 
        self.slider_pe_ratio = max([float(stock.pe_ratio) for stock in self.thread.stocks])
        self.slider_performance_52_week = max([float(stock.performance_52_week) for stock in self.thread.stocks])
        self.slider_performance_52_week_lower = min([float(stock.performance_52_week) for stock in self.thread.stocks])
        self.slider_average_volume_10day = max([stock.average_volume_10day for stock in self.thread.stocks])
        self.slider_average_volume_10day_lower = min([stock.average_volume_10day for stock in self.thread.stocks])
        self.slider_obv = max([stock.obv for stock in self.thread.stocks])
        self.slider_adx = max([stock.adx for stock in self.thread.stocks])
        self.slider_macd = max([stock.macd for stock in self.thread.stocks])
        self.slider_macd_lower = min([stock.macd for stock in self.thread.stocks])
        self.slider_number_of_new_highs = max([stock.number_of_new_highs for stock in self.thread.stocks])
        
        score_filtered_stocks = self.thread.stocks
        score_filtered_stocks.sort(key=lambda stock: stock.weighted_score, reverse=True)
        for i, stock in enumerate(score_filtered_stocks):
            if i < 100:
                self.top_scores.append(stock.symbol)
                i += 1
            else:
                break

        self.create_widgets()
        self.show()

    def start_loading_thread(self):
        self.thread = LoadingThread(self.ticker_list, [], self, self.loading_window, start_date=self.start_date, end_date=self.end_date)
        self.thread.done.connect(self.loading_done)
        self.thread.progress.connect(self.loading_window.update_progress)
        self.thread.start()

        self.loading_window.show()

    def set_stock_screener(self, stock_screener):
        self.stock_screener = stock_screener

    def save_settings(self):
        settings = {
            "min_market_cap": self.min_market_cap.text(),
            "min_dividend_yield": self.min_dividend_yield.text(),
            "max_pe_ratio": self.max_pe_ratio.text(),
            "min_performance_52_week": self.min_performance_52_week.text(),
            "min_obv": self.min_obv.text(),
            "min_adx": self.min_adx.text(),
            "sector_filter": self.sector_filter.currentText(),
            "volume_filter": self.volume_filter.text(),
            "macd_filter": self.macd_filter.text(),
            "number_of_new_highs": self.number_of_new_highs.text()
        }
        with open("settings.json", "w") as file:
            json.dump(settings, file)

    def load_settings(self):
        try:
            with open("settings.json", "r") as file:
                settings = json.load(file)
            self.min_market_cap.setText(settings["min_market_cap"])
            self.min_dividend_yield.setText(settings["min_dividend_yield"])
            self.max_pe_ratio.setText(settings["max_pe_ratio"])
            self.min_performance_52_week.setText(settings["min_performance_52_week"])
            self.min_obv.setText(settings["min_obv"])
            self.min_adx.setText(settings["min_adx"])
            index = self.sector_filter.findText(settings["sector_filter"])
            if index >= 0:
                self.sector_filter.setCurrentIndex(index)
            self.volume_filter.setText(settings["volume_filter"])
            self.macd_filter.setText(settings["macd_filter"])
            self.number_of_new_highs.setText(settings["number_of_new_highs"])
            
        except FileNotFoundError:
            QMessageBox.critical(self, "Error", "No saved settings found.")  

    def closeEvent(self, event):
        event.accept()
        
    def create_widgets(self):
        #Date range selection
        self.layout.addWidget(QLabel("Selected Date Range:"))
        self.layout.addWidget(QLabel(self.start_date))
        self.layout.addWidget(QLabel(self.end_date))

        # Create input fields
        sectors = ["","Healthcare", "Industrials", "Technology", "Financial Services", 
                   "Consumer Cyclical", "Energy", "Consumer Defensive", 
                   "Communication Services", "Basic Materials","Real Estate"]
        self.sector_filter = QComboBox()
        self.sector_filter.addItems(sectors)
        self.layout.addWidget(QLabel("Sector:"))
        self.layout.addWidget(self.sector_filter)

        self.add_numeric_filter("Min Market Cap(in millions):", 0, self.slider_max_market_cap, 1, self.min_market_cap, self.min_market_cap_slider)
        self.add_numeric_filter("Min Dividend Yield:", 0, self.slider_div_yield, 100, self.min_dividend_yield, self.min_dividend_yield_slider)
        self.add_numeric_filter("Max P/E Ratio:", 0, self.slider_pe_ratio, 1, self.max_pe_ratio, self.max_pe_ratio_slider)
        self.add_numeric_filter("Min 52-Week Performance:", 0, self.slider_performance_52_week, 1, self.min_performance_52_week, self.min_performance_52_week_slider)
        self.add_numeric_filter("Minimum average 10 day volume:", 0, self.slider_average_volume_10day, 1, self.volume_filter, self.volume_filter_slider)
        self.add_numeric_filter("Min on balance value:", 0, self.slider_obv, 1, self.min_obv, self.min_obv_slider)
        self.add_numeric_filter("Min average directional index:", 0, self.slider_adx, 1, self.min_adx, self.min_adx_slider)
        self.add_numeric_filter("MACD:", 0, self.slider_macd, 1, self.macd_filter, self.macd_filter_slider)
        self.add_numeric_filter("Number of New Highs:", 0, self.slider_number_of_new_highs, 1, self.number_of_new_highs, self.number_of_new_highs_slider)

        """self.layout.addWidget(QLabel("Min Market Cap(in millions):"))
        self.layout.addWidget(self.min_market_cap)

        self.layout.addWidget(QLabel("Min Dividend Yield:"))
        self.layout.addWidget(self.min_dividend_yield)

        self.layout.addWidget(QLabel("Max P/E Ratio:"))
        self.layout.addWidget(self.max_pe_ratio)

        self.layout.addWidget(QLabel("Min 52-Week Performance:"))
        self.layout.addWidget(self.min_performance_52_week)

        self.layout.addWidget(QLabel("Minimum average 10 day volume:"))
        self.layout.addWidget(self.volume_filter)

        self.layout.addWidget(QLabel("Min on balance value:"))
        self.layout.addWidget(self.min_obv)

        self.layout.addWidget(QLabel("Min average directional index:"))
        self.layout.addWidget(self.min_adx)

        self.layout.addWidget(QLabel("MACD:"))
        self.layout.addWidget(self.macd_filter)

        self.layout.addWidget(QLabel("Number of New Highs:"))
        self.layout.addWidget(self.number_of_new_highs)
        """
        self.new_highs = QCheckBox("At New High Only", self)
        self.layout.addWidget(self.new_highs)

        # Create filter button
        self.filter_button = QPushButton("Filter", self)
        self.filter_button.clicked.connect(self.filter_stocks)
        self.layout.addWidget(self.filter_button)

        # Create save button
        self.save_button = QPushButton("Save Settings", self)
        self.save_button.clicked.connect(self.save_settings)
        self.layout.addWidget(self.save_button)

        # Create load button
        self.load_button = QPushButton("Load Settings", self)
        self.load_button.clicked.connect(self.load_settings)
        self.layout.addWidget(self.load_button)

        # Create clear button
        self.clear_button = QPushButton("Clear All Inputs", self)
        self.clear_button.clicked.connect(self.clear_fields)
        self.layout.addWidget(self.clear_button)

        # Create Treeview
        self.tree = QTableWidget(self)
        self.tree.setRowCount(0)
        self.tree.setColumnCount(15)
        self.tree.setHorizontalHeaderLabels(["Symbol", "Sector", "Market Cap(in millions)", "Dividend Yield", "P/E Ratio", "52-Week Performance","MinimumAverage10DayVolume","On Balance Value","Min average directional index","MACD","Number of New Highs in Date Range","macd buy signal (in the most recent trading session)","MACD Histogram", "Weighted Score", "Advisor result"])
        self.tree.setSortingEnabled(True)
        self.splitter.addWidget(self.tree)

    def clear_fields(self):
        # Clear QLineEdit widgets
        self.min_market_cap.clear()
        self.min_dividend_yield.clear()
        self.max_pe_ratio.clear()
        self.min_performance_52_week.clear()
        self.volume_filter.clear()
        self.min_obv.clear()
        self.min_adx.clear()
        self.macd_filter.clear()
        self.number_of_new_highs.clear()

        # Clear QSlider widgets
        self.min_market_cap_slider.setValue(0)
        self.min_dividend_yield_slider.setValue(0)
        self.max_pe_ratio_slider.setValue(0)
        self.min_performance_52_week_slider.setValue(0)
        self.volume_filter_slider.setValue(0)
        self.min_obv_slider.setValue(0)
        self.min_adx_slider.setValue(0)
        self.macd_filter_slider.setValue(0)
        self.number_of_new_highs_slider.setValue(0)

        # Clear QComboBox
        self.sector_filter.setCurrentIndex(0)  # Assuming the first item is the default one

        # Uncheck the checkbox
        self.new_highs.setChecked(False)

        # clear the results shown in the QTableWidget
        self.tree.setRowCount(0)
        
    def filter_stocks(self):
        min_market_cap = self.get_float(self.min_market_cap.text())
        min_dividend_yield = self.get_float(self.min_dividend_yield.text())
        max_pe_ratio = self.get_float(self.max_pe_ratio.text())
        min_performance_52_week = self.get_float(self.min_performance_52_week.text())
        min_obv = self.get_float(self.min_obv.text())
        min_adx = self.get_float(self.min_adx.text())
        sector_filter = self.sector_filter.currentText() if self.sector_filter.currentText() else None
        volume_filter = self.get_float(self.volume_filter.text())
        macd_filter = self.get_float(self.macd_filter.text())
        new_highs = self.new_highs.isChecked()
        number_of_new_highs = self.get_float(self.number_of_new_highs.text())

        filtered_stocks = self.stock_screener.filter_stocks(
            min_market_cap, 
            min_dividend_yield, 
            max_pe_ratio, 
            min_performance_52_week, 
            min_obv, 
            min_adx,
            sector_filter,
            volume_filter,
            macd_filter, 
            new_highs,
            number_of_new_highs
        )
        self.filtered_stocks_list = [stock.symbol for stock in filtered_stocks]
        self.update_tree(filtered_stocks)

    def update_tree(self, stocks):
        self.tree.setRowCount(0)
        for i, stock in enumerate(stocks):
            self.tree.insertRow(i)
            self.tree.setItem(i, 0, QTableWidgetItem(stock.symbol))
            self.tree.setItem(i, 1, QTableWidgetItem(stock.sector))
            self.tree.setItem(i, 2, NumericTableWidgetItem(str(stock.market_cap)))
            self.tree.setItem(i, 3, NumericTableWidgetItem(str(stock.dividend_yield)))
            self.tree.setItem(i, 4, NumericTableWidgetItem(str(stock.pe_ratio)))
            self.tree.setItem(i, 5, NumericTableWidgetItem(str(stock.performance_52_week)))
            self.tree.setItem(i, 6, NumericTableWidgetItem(str(stock.average_volume_10day)))
            self.tree.setItem(i, 7, NumericTableWidgetItem(str(stock.obv)))
            self.tree.setItem(i, 8, NumericTableWidgetItem(str(stock.adx)))
            self.tree.setItem(i, 9, NumericTableWidgetItem(str(stock.macd)))
            self.tree.setItem(i, 10, NumericTableWidgetItem(str(stock.number_of_new_highs)))
            self.tree.setItem(i, 11, QTableWidgetItem(str(stock.macd_buy)))

            # MACD Histogram button
            btn = QPushButton('Show MACD Histogram')
            btn.clicked.connect(lambda checked, x=stock: self.show_hist(x))
            self.tree.setCellWidget(i, 12, btn)
            self.tree.setItem(i, 13, NumericTableWidgetItem(str(stock.weighted_score)))
            self.tree.setItem(i, 14, QTableWidgetItem(str(stock.advisor_result)))

    
    def show_hist(self, stock):
        hist = stock.macd_hist_data
        plt.figure(num='',figsize=(14, 7))
        plt.plot(hist, label='MACD Histogram')
        plt.title('MACD Histogram for ' + stock.symbol)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    # convert to float so I can compare the input with the data
    @staticmethod
    def get_float(value):
        try:
            return float(value)
        except ValueError:
            return None

    def close_app(self):
        self.close()

# This function is responsible for retriving data from the yahoo finnace library, anything not present in the 
# data base are set to 0 instead.
class LoadingThread(QThread):
    # Define a signal to emit when done
    progress = pyqtSignal(int)
    done = pyqtSignal()

    def __init__(self, ticker_list, stocks, app, loading_window,start_date=None, end_date=None, max_tickers=100):
        super().__init__()
        self.ticker_list = ticker_list
        self.stocks = stocks
        self.app = app
        self.loading_window = loading_window
        self.start_date = start_date
        self.end_date = end_date
        self.max_tickers = max_tickers
        self.model_dir = "models"

    def run(self):
        self.load_stock_data()
        print("The number of tickers in this sample is: ",len(self.stocks))
        self.done.emit()

    def load_stock_data(self):
        i = 0
        for symbol in self.ticker_list:
            #if i >= self.max_tickers:
                #break
            try:
                stock = self.get_stock_data(symbol, self.start_date, self.end_date)
                if stock is not None:
                    self.stocks.append(stock)
                    i += 1
                    self.progress.emit(int(i / len(self.ticker_list) * 100))
                    #self.progress.emit(int(i / self.max_tickers * 100))
            except Exception as e:
                print(f'Error loading data for {symbol}: {e}')
                continue
    
    def get_stock_data(self, symbol, start_date=None, end_date=None):
        ticker = yf.Ticker(symbol)
        stock_info = ticker.info
        if start_date and end_date:
            # Fetch data for the specified date range
            stock_history = ticker.history(start=start_date, end=end_date)
        else:
            print("default 3mo used")
            # Fetch 3 months of data by default
            stock_history = ticker.history(period="3mo")
        sector = stock_info.get("sector", "")
        market_cap = stock_info.get("marketCap", 0) / 1e6  # Convert to millions
        dividend_yield = stock_info.get("dividendYield", 0)
        pe_ratio = stock_info.get("trailingPE", 0)
        performance_52_week = stock_info.get("52WeekChange", 0)
        average_volume_10day = stock_info.get("averageVolume10days", 0)
        new_high = stock_history['High'].max()
        current_price = stock_history.iloc[-1]['Close']
        is_new_high = current_price >= new_high
        
        # OBV calculation
        close_prices = stock_history['Close']
        volume = stock_history['Volume']
        obv = talib.OBV(close_prices, volume)

        # ADX calculation
        high_prices = stock_history['High']
        low_prices = stock_history['Low']
        adx = talib.ADX(high_prices, low_prices, close_prices)

        # Calculate MACD
        macd, signal, hist = talib.MACD(close_prices)

        
        # Recency score
        macd_buy_signals = (macd > signal) & (macd.shift(1) < signal.shift(1))
        recency_weights = np.arange(1,0,-0.2)
        weighted_buy_signals = macd_buy_signals[-len(recency_weights):] * recency_weights
        recency_score = weighted_buy_signals.sum()
        #normalization
        max_possible_recency_score = recency_weights.sum()
        recency_score = recency_score / max_possible_recency_score

        # Normalize adx to a score
        adx_values = adx.dropna().reset_index(drop=True)
        adx_weights = np.arange(len(adx_values)) / len(adx_values)
        adx_weights = adx_weights[::-1]
        adx_score = np.average(adx_values, weights=adx_weights)
        normalized_adx_score = adx_score / 100

        # Total Score
        total_score = 0.4 * recency_score + 0.6 * normalized_adx_score

        # Find where MACD crosses above the signal line
        macd_buy = (macd[-1] > signal[-1]) and (macd[-2] < signal[-2])

        # Calculate number of new highs
        number_of_new_highs = 0
        for i in range(1, len(high_prices)):
            if high_prices[i] > high_prices[:i].max():
                number_of_new_highs += 1

        # Invoke subprocess/advisor and get results
        pass_value = symbol
        print("passing to subprocess", symbol)
        result = subprocess.run(["python", "main.py", pass_value], check=True, text=True, stdout=subprocess.PIPE)
        print("value passed:",result.stdout)

        variables = [sector, market_cap, dividend_yield, pe_ratio, performance_52_week, average_volume_10day, new_high, current_price, obv, adx, macd, signal, hist]
        if any(v is None for v in variables):
            print(f'Skipping {symbol} due to missing data')
            return None
        return Stock(symbol, sector, market_cap, dividend_yield, pe_ratio, performance_52_week, obv, adx, average_volume_10day, macd, signal, macd_buy, is_new_high, number_of_new_highs, hist, total_score, result)

class DateRange(QMainWindow):
    datesSelected = pyqtSignal(str, str)
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Date Range Selection")
        self.setGeometry(100, 100, 400, 200)

        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDate(QDate.currentDate().addMonths(-3))

        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDate(QDate.currentDate())

        self.load_data_button = QPushButton("Load Data")
        self.load_data_button.clicked.connect(self.load_data)

        layout.addWidget(self.start_date)
        layout.addWidget(self.end_date)
        layout.addWidget(self.load_data_button)

    def load_data(self):
        start_date = self.start_date.date().toString(Qt.DateFormat.ISODate)
        end_date = self.end_date.date().toString(Qt.DateFormat.ISODate)

        self.datesSelected.emit(start_date, end_date)
        self.close()

# Test main method, currently it gets a list of all nasdaq tickers from a datahub csv file I found online
# going through the entire list takes the program roughly 10-15 mins so I limited the sample size to 100 for now
# Filtering is done in the temperary UI and after filtering, you can close the window and the produced ticker list
# is stored in the app.filtered_stocks_list variable for the RoboAdvisor to use.
def main():
    app = QApplication([])
    # Load ticker list from prefilted list
    ticker_list = []
    try:
        with open('filtered_stocks.csv', 'r') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                ticker_list.append(row[0]) 
    except FileNotFoundError:
        print("File 'filtered_stocks.csv' not found.")

    # Initialize app and run
    screener_app = StockScreenerApp(ticker_list)
    screener_app.retrieve_date()
    app.exec()

    print(screener_app.filtered_stocks_list)
    print("Top 100 scores stored in here:" ,screener_app.top_scores)
    df = pd.DataFrame(screener_app.top_scores, columns=['Ticker'])
    df.to_csv('Top_Score_Output.csv', index = False)

if __name__ == '__main__':
    main()
