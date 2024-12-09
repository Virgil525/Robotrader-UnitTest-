import pandas as pd
import talib
import numpy as np
import yfinance as yf
import json
import csv
from PyQt6.QtCore import QThread, pyqtSignal, QDate, Qt, QEventLoop, pyqtSignal
from PyQt6.QtWidgets import QScrollArea, QSplitter, QSlider, QDateEdit, QComboBox, QMessageBox, QTableWidgetItem, QTableWidget, QPushButton, QMainWindow, QLabel, QVBoxLayout, QProgressBar, QWidget, QApplication, QLineEdit, QCheckBox
import matplotlib.pyplot as plt
#import main
import asyncio
from ibapi.client import *
from ibapi.wrapper import *
from ib_insync import *
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import nest_asyncio
from bs4 import BeautifulSoup as bs

# AWS Secrets
#import boto3
#import base64
#from botocore.exceptions import BotoCoreError, ClientError
"""
def get_secret():

    access_key_id = 'AKIA56VSHTRNJW3RLFNM'
    secret_access_key = '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'
    region_name = 'us-east-2'
    secret_name = 'IB_Credential_VGL'
    
    print("Starting to get secret...") # for debugging
    
    session = boto3.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region_name
    )
    
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    secrets_json = None
    try:
        print("About to call get_secret_value...")  # for debugging
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )

        print(get_secret_value_response)
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            secrets_json = json.loads(secret)
        elif 'SecretBinary' in get_secret_value_response:
            secret = base64.b64decode(get_secret_value_response['SecretBinary'])
            secrets_json = json.loads(secret)
        else:
            raise Exception("Neither 'SecretString' nor 'SecretBinary' found in the response.")

    except ClientError as e:
        if e.response['Error']['Code'] in ['DecryptionFailureException',
                                            'InternalServiceErrorException',
                                            'InvalidParameterException',
                                            'InvalidRequestException',
                                            'ResourceNotFoundException']:
            print(f"Exception occurred: {e}")  # for debugging
            raise e
    print(get_secret_value_response)
    print("Finished getting secret.")  # for debugging
    return secrets_json['username'], secrets_json['password']

"""

# Apply nest_asyncio to enable nested usage of asyncio's event loop, Reason: IBAPI use asyncio too, will raise exception
nest_asyncio.apply()

# APP stuff
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
        self.slider_max_market_cap = max([float(stock.market_cap) for stock in self.thread.stocks])
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
        self.layout.addWidget(QLabel(self.start_date.strftime('%Y-%m-%d %H:%M:%S')))
        self.layout.addWidget(QLabel(self.end_date.strftime('%Y-%m-%d %H:%M:%S')))

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

    def __init__(self, ticker_list, stocks, app, loading_window, start_date=None, end_date=None, max_tickers=100):
        super().__init__()
        self.ticker_list = ticker_list
        self.stocks = stocks
        self.app = app
        self.loading_window = loading_window
        self.start_date = start_date
        self.end_date = end_date
        self.max_tickers = max_tickers

        self.ib = IB()

    def run(self):
        # Define a new event loop
        loop = asyncio.new_event_loop()
        # Set the event loop for the current context
        asyncio.set_event_loop(loop)

        # Load stock data within the event loop
        loop.run_until_complete(self.load_stock_data())
        print("The number of tickers in this sample is: ",len(self.stocks))

        self.done.emit()

    async def load_stock_data(self):
        # Connect to IB
        #await self.ib.connectAsync('127.0.0.1', 7497, clientId=1)
        await self.ib.connectAsync('127.0.0.1', 4001, clientId=1)
        i = 0
        for symbol in self.ticker_list:
            # if i >= self.max_tickers:
            #     break
            try:
                stock = await self.get_stock_data(symbol, self.start_date, self.end_date)
                i += 1
                if stock is not None:
                    self.stocks.append(stock)
                    #i += 1
                    #self.progress.emit(int(i / len(self.ticker_list) * 100))

                    # self.progress.emit(int(i / self.max_tickers * 100))
                self.progress.emit(int(i / len(self.ticker_list) * 100))
            except Exception as e:
                print(f'Error loading data for {symbol}: {e}')
                continue
        self.ib.disconnect()

    def get_fundamental_data(self, xml_data):
        root = ET.fromstring(xml_data)
    
        symbol = root.find(".//Issues/Issue/IssueID[@Type='Ticker']").text
        sector = root.find(".//peerInfo/IndustryInfo/Industry[@type='TRBC']").text
        market_cap = root.find(".//Ratios/Group[@ID='Income Statement']/Ratio[@FieldName='MKTCAP']").text
        
        dividend_per_share = float(root.find(".//Ratios/Group[@ID='Per share data']/Ratio[@FieldName='TTMDIVSHR']").text)
        stock_price = float(root.find(".//Ratios/Group[@ID='Price and Volume']/Ratio[@FieldName='NPRICE']").text)
        dividend_yield = (dividend_per_share / stock_price) * 100
        
        pe_ratio = root.find(".//Ratios/Group[@ID='Other Ratios']/Ratio[@FieldName='PEEXCLXOR']").text
        
        fundamental_data = {
            'symbol': symbol,
            'sector': sector,
            'market_cap': market_cap,
            'dividend_yield': dividend_yield,
            'pe_ratio': pe_ratio
        }
        #print(fundamental_data)
        return fundamental_data

    async def get_stock_data(self, symbol, start_date=None, end_date=None):
        # Set up the contract
        contract = Contract(symbol=symbol, secType='STK', exchange='SMART', currency='USD')
        await self.ib.qualifyContractsAsync(contract)

        # Define the time period for historical data
        if not start_date:
            start_date = datetime.now() - timedelta(days=90)  # default to past 3 months
        if not end_date:
            end_date = datetime.now()

        duration = end_date - start_date

         # Get historical data
        bars = await self.ib.reqHistoricalDataAsync(
            contract,
            endDateTime=end_date,
            durationStr=str(duration.days) + ' D',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True,
            formatDate=1
        )

        if bars:
            df = util.df(bars)
            df.set_index('date', inplace=True)
        else:
            print(f"Couldn't retrieve data for {symbol}")
            return None

        # Get fundamental data
        try:
            xml_data = await self.ib.reqFundamentalDataAsync(contract, 'ReportSnapshot')
            #print(f"Raw XML data for {symbol}: {xml_data}")
            fundamental_data = self.get_fundamental_data(xml_data)
        except Exception as e:
            print(f"Couldn't retrieve fundamental data for {symbol}: {e}")
            fundamental_data = {}

        # key debug
        if 'sector' not in fundamental_data:
            print(f"Fundamental data for {symbol}: {fundamental_data}")

        # temp handling for fundamental data
        sector = fundamental_data.get('sector', 'Unknown')
        market_cap = fundamental_data.get('market_cap', 0)
        dividend_yield = fundamental_data.get('dividend_yield', 0)
        pe_ratio = fundamental_data.get('pe_ratio', 0)


        # Calculate technical indicators: OBV, ADX, MACD
        obv = talib.OBV(df['close'].values, df['volume'].values)
        adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values)
        macd, macd_signal, macd_hist_data = talib.MACD(df['close'].values)

        # Array aligning
        # Check alignment and compare MACD with MACD signal
        start_idx = np.where(~np.isnan(macd_signal))[0][0]
        # Ensure that aligned_macd and aligned_macd_signal have the same length
        end_idx = min(len(macd) - start_idx, len(macd_signal) - start_idx)
        aligned_macd = macd[start_idx: start_idx + end_idx]
        aligned_macd_signal = macd_signal[start_idx: start_idx + end_idx]
        
        # MACD Buy signals
        macd_buy_signals = (aligned_macd[1:] > aligned_macd_signal[:-1]) & (aligned_macd[:-1] < aligned_macd_signal[:-1])
        recent_macd_buy_signal = macd_buy_signals[-1]

        # Recency score
        #macd_comparison = (macd > macd_signal)[:-1]  # Compare and then trim the last value
        #macd_shifted_comparison = (macd[:-1] < macd_signal[:-1])
        #macd_buy_signals = (aligned_macd > aligned_macd_signal) & (aligned_macd[:-1] < aligned_macd_signal[:-1])
        #macd_buy_signals = (macd > macd_signal) & (macd[:-1] < macd_signal[:-1])
        
        recency_weights = np.arange(1,0,-0.2)
        weighted_buy_signals = macd_buy_signals[-len(recency_weights):] * recency_weights
        recency_score = np.sum(weighted_buy_signals)
        #normalization
        max_possible_recency_score = np.sum(recency_weights)
        recency_score = recency_score / max_possible_recency_score

        # Normalize adx to a score
        adx_values = adx[np.logical_not(np.isnan(adx))]
        adx_weights = np.arange(len(adx_values)) / len(adx_values)
        adx_weights = adx_weights[::-1]
        adx_score = np.average(adx_values, weights=adx_weights)
        normalized_adx_score = adx_score / 100

        # Total Score
        total_score = 0.4 * recency_score + 0.6 * normalized_adx_score

        """        # Find where MACD crosses above the signal line
        if len(macd) == len(macd_signal):
            macd_buy = (macd[-1] > macd_signal[-1]) and (macd[-2] < macd_signal[-2])
        else:
            print(f"Length mismatch in MACD data for {symbol}: macd({len(macd)}) vs macd_signal({len(macd_signal)})")
            macd_buy = False  # Default value"""

        # Calculate number of new highs
        number_of_new_highs = 0
        if len(df['high']) > 1:
            for i in range(1, len(df['high'])):
                if df['high'].iloc[i] > df['high'].iloc[:i].max():
                    number_of_new_highs += 1

        # Initialize Stock object
        stock = Stock(
            symbol=symbol,
            sector=sector,
            market_cap=market_cap,
            dividend_yield=dividend_yield,
            pe_ratio=pe_ratio,
            performance_52_week=df['close'].pct_change(periods=52).iloc[-1], 
            obv=obv,
            adx=adx,
            average_volume_10day=df['volume'].rolling(window=10).mean().iloc[-1],
            macd=macd,
            macd_signal=macd_signal,
            macd_buy=recent_macd_buy_signal,
            new_high=df['high'].max(),
            number_of_new_highs=(df['high'] > df['high'].shift()).sum(),
            macd_hist_data=macd_hist_data,
            total_score=total_score, 
            result=1  # Not used in this method
        )

        return stock

class DateRange(QMainWindow):
    datesSelected = pyqtSignal(datetime, datetime)
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
        start_qdate = self.start_date.date()
        start_date = datetime(start_qdate.year(), start_qdate.month(), start_qdate.day())

        end_qdate = self.end_date.date()
        end_date = datetime(end_qdate.year(), end_qdate.month(), end_qdate.day())

        self.datesSelected.emit(start_date, end_date)
        self.close()

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

    #username, password = get_secret()
    #print(username, password)
    # Initialize app and run
    screener_app = StockScreenerApp(ticker_list)
    screener_app.retrieve_date()
    app.exec()
    
    #print(screener_app.filtered_stocks_list)
    print("Top 100 scores stored in here:" ,screener_app.top_scores)
    #df = pd.DataFrame(screener_app.top_scores, columns=['Ticker'])
    #df.to_csv('Top_Score_Output.csv', index = False)

if __name__ == '__main__':
    main()
