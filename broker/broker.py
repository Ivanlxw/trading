from typing import List
import datetime
import os
import threading
import time
import logging
from abc import ABC, abstractmethod
import requests
import pandas as pd
import json

from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.wrapper import EWrapper
import alpaca_trade_api

from backtest.utilities.utils import log_message
from trading.broker.gatekeepers import DummyGateKeeper, GateKeeper
from trading.event import FillEvent, OrderEvent
from trading.portfolio.portfolio import NaivePortfolio, Portfolio
from trading.utilities.enum import OrderPosition, OrderType


class Broker(ABC):
    @abstractmethod
    def _filter_execute_order(self, event: OrderEvent) -> bool:
        """
        Takes an OrderEvent and checks if it can be executed
        """
        raise NotImplementedError("Should implement _filter_execute_order()")

    def execute_order(self, event) -> bool:
        """
        Takes an OrderEvent and execute it, producing
        FillEvent that gets places into Events Queue
        """
        raise NotImplementedError("Should implement execute_order()")

    @abstractmethod
    def calculate_commission(self,):
        """
        Takes an OrderEvent and calculates commission based 
        on the details of the order it, add as argument for 
        FillEvent
        """
        raise NotImplementedError("Implement calculate_commission()")

    def check_filled_order(self):
        """ More for live trading, since we send q request with no response from web api, we need to check via API for pending & filled orders"""

    def update_portfolio_positions(self, port: Portfolio):
        """ Update for alpaca and TDA, use API call to update Portfolio class positions"""

    def get_account_details(self):
        """ For live brokers """

"""
TODO: 
- include slippage and market impact
- make use of the "current" market data value to obtain a realistic fill cost.
"""


class SimulatedBroker(Broker):
    def __init__(self, bars, port, events, order_queue, gatekeepers: List[GateKeeper] = None):
        self.bars = bars
        self.port: NaivePortfolio = port
        self.events = events
        self.order_queue = order_queue
        if gatekeepers is None or len(gatekeepers) == 0:
            self.gatekeepers = [DummyGateKeeper()]
        else:
            self.gatekeepers = gatekeepers

    def calculate_commission(self, quantity=None, fill_cost=None) -> float:
        return 0.0

    def _order_expired(self, latest_ohlc, order_event: OrderEvent):
        # check for expiry
        return latest_ohlc["datetime"][-1] > order_event.expires

    def _filter_execute_order(self, latest_snapshot, order_event: OrderEvent) -> bool:
        if order_event.order_type == OrderType.LIMIT:
            """                
            print("Price data:", price_data_dict)
            print("OrderEvent: ", {
                "Symbol": event.symbol, 
                "Price": event.signal_price, 
                "Direction": event.direction,
                "Expires": event.expires
            })
            """
            if (
                order_event.signal_price > latest_snapshot["high"][-1]
                and order_event.direction == OrderPosition.BUY
            ) or (
                order_event.signal_price < latest_snapshot["low"][-1]
                and order_event.direction == OrderPosition.SELL
            ):
                return False
        return True

    def _put_fill_event(self, order_event: OrderEvent):
        order_event.trade_price = order_event.signal_price
        fill_event = FillEvent(order_event, self.calculate_commission())
        self.events.put(fill_event)
        return True

    def execute_order(self, event: OrderEvent) -> bool:
        if event is None:
            return False
        if event.type == "ORDER" and all(gk.check_gk(event, self.port.current_holdings) for gk in self.gatekeepers):
            for gk in self.gatekeepers:
                event = gk._alter_order(event, self.port.current_holdings)
            if event is not None:
                latest_snapshot = self.bars.get_latest_bars(event.symbol)
                if event.order_type == OrderType.LIMIT:
                    if not self._order_expired(latest_snapshot, event):
                        if not event.processed:
                            event.processed = True
                            event.date += datetime.timedelta(days=1)
                            self.order_queue.put(event)
                            return False
                        if self._filter_execute_order(latest_snapshot, event):
                            return self._put_fill_event(event)
                        event.date += datetime.timedelta(days=1)
                        self.order_queue.put(event)
                elif event.order_type == OrderType.MARKET:
                    event.trade_price = latest_snapshot["close"][-1]
                    return self._put_fill_event(event)
            return False
        return False


class IBBroker(Broker, EWrapper, EClient):
    def __init__(self, events):
        EClient.__init__(self, self)
        self.events = events
        self.fill_dict = {}
        self.hist_data = []

        self.connect("127.0.0.1", 7497, 123)
        self.tws_conn = self.create_tws_connection()
        self.reqMarketDataType(3)  # DELAYED
        self.order_id = self.create_initial_order_id()
        # self.register_handlers()

    def create_tws_connection(self) -> None:
        """
        Connect to the Trader Workstation (TWS) running on the
        usual port of 7496, with a clientId of 10.
        The clientId is chosen by us and we will need 
        separate IDs for both the broker connection and
        market data connection, if the latter is used elsewhere.
        - Should run in a new thread
        """

        def run_loop():
            self.run()

        api_thread = threading.Thread(target=run_loop, daemon=True)
        api_thread.start()
        time.sleep(1)  # to allow connection to server

    def create_initial_order_id(self):
        """
        Creates the initial order ID used for Interactive
        Brokers to keep track of submitted orders.

        Can always reset the current API order ID via: 
        Trader Workstation > Global Configuration > API Settings panel:
        """
        # will use "1" as the default for now.
        return 1

    # create a Contract instance and then pair it with an Order instance,
    # which will be sent to the IB API
    def create_contract(self, symbol, sec_type, exchange="SMART", currency="USD"):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = sec_type
        contract.exchange = exchange
        contract.currency = currency
        return contract

    # Overwrite EClient.historicalData
    def historicalData(self, reqId, bar):
        print(f"Time: {bar.date} Close: {bar.close}")
        self.hist_data.append([bar.date, bar.close])
        # if eurusd_contract.symbol in my_td_broker.hist_data.keys():
        #     my_td_broker.hist_data[eurusd_contract.symbol].append()

    def _error_handler(self, msg):
        # Handle open order orderId processing
        if (
            msg.typeName == "openOrder"
            and msg.orderId == self.order_id
            and not self.fill_dict.has_key(msg.orderId)
        ):
            self.create_fill_dict_entry(msg)
        # Handle Fills
        elif (
            msg.typeName == "orderStatus"
            and msg.status == "Filled"
            and self.fill_dict[msg.orderId]["filled"] == False
        ):
            self.create_fill(msg)
        print("Server Response: {}, {}\n".format(msg.typeName, msg))

    def register_handlers(self):
        """
        Register the error and server reply  message handling functions.
        """
        # Assign the error handling function defined above
        # to the TWS connection
        self.tws_conn.register(self._error_handler, "Error")

        # Assign all of the server reply messages to the
        # reply_handler function defined above
        self.tws_conn.registerAll(self._reply_handler)

    def create_order(self, order_type, quantity, action):
        """
            order_type - MARKET, LIMIT for Market or Limit orders
            quantity - Integral number of assets to order
            action - 'BUY' or 'SELL'
        """
        order = OrderEvent()
        order.m_orderType = order_type
        order.m_totalQuantity = quantity
        order.m_action = action

        return order

    def create_fill_dict_entry(self, msg):
        """
        needed for the event-driven behaviour of the IB
        server message behaviour.
        """
        self.fill_dict[msg.orderId] = {
            "symbol": msg.contract.m_symbol,
            "exchange": msg.contract.m_exchange,
            "direction": msg.order.m_action,
            "filled": False,
        }

    def create_fill(self, msg):
        fd = self.fill_dict[msg.orderId]
        symbol = fd["symbol"]
        exchange = fd["exchange"]
        filled = msg.filled
        direction = fd["direction"]
        fill_cost = msg.avgFillPrice

        fill = FillEvent(
            datetime.datetime.utcnow(), symbol, exchange, filled, direction, fill_cost
        )

        self.fill_dict[msg.orderId]["filled"] = True

        self.events.put(fill)

    def execute_order(self, event) -> bool:
        if event is None:
            return False
        if event.type == "ORDER" and all(gk.check_gk(event, self.port.current_holdings) for gk in self.gatekeepers):
            for gk in self.gatekeepers:
                event = gk._alter_order(event, self.port.current_holdings)
            if event is not None:
                asset = event.symbol
                asset_type = "STK"
                order_type = event.order_type
                quantity = event.quantity
                direction = event.direction

                # Create the Interactive Brokers contract via the
                # passed Order event
                ib_contract = self.create_contract(
                    asset, asset_type, self.order_routing, self.order_routing, self.currency
                )

                # Create the Interactive Brokers order via the
                # passed Order event
                ib_order = self.create_order(order_type, quantity, direction)

                # Use the connection to the send the order to IB
                self.tws_conn.placeOrder(self.order_id, ib_contract, ib_order)

                time.sleep(1)
                self.order_id += 1
                return True
        return False

    def calculate_commission(self, quantity, fill_cost):
        full_cost = 1.3
        if quantity <= 500:
            full_cost = max(1.3, 0.013 * quantity)
        else:
            full_cost = max(1.3, 0.008 * quantity)
        full_cost = min(full_cost, 0.005 * quantity * fill_cost)

        return full_cost


class TDABroker(Broker):
    def __init__(self, events, gatekeepers: List[GateKeeper]) -> None:
        super(TDABroker, self).__init__()
        self.events = events
        self.consumer_key = os.environ["TDD_consumer_key"]
        self.account_id = os.environ["TDA_account_id"]
        self.access_token = None
        self.refresh_token = None
        self.get_token("authorization")
        if gatekeepers is None or len(gatekeepers) == 0:
            self.gatekeepers = [DummyGateKeeper()]
        else:
            self.gatekeepers = gatekeepers
        self.pending_orders: List[OrderEvent] = []

    def _signin_code(self):
        import selenium
        from selenium import webdriver
        from backtest.utilities.utils import load_credentials, parse_args

        args = parse_args()
        load_credentials(args.credentials)

        driver = webdriver.Firefox()
        url = f"https://auth.tdameritrade.com/auth?response_type=code&redirect_uri=http://localhost&client_id={self.consumer_key}@AMER.OAUTHAP"
        driver.get(url)

        userId = driver.find_element_by_css_selector("#username0")
        userId.clear()
        userId.send_keys(os.environ["TDA_username"])
        pw = driver.find_element_by_css_selector("#password1")
        pw.clear()
        pw.send_keys(f"{os.environ['TDA_pw']}")
        login_button = driver.find_element_by_css_selector("#accept")
        login_button.click()

        # click accept
        accept_button = driver.find_element_by_css_selector("#accept")
        try:
            accept_button.click()
        except selenium.common.exceptions.WebDriverException:
            new_url = driver.current_url
            code = new_url.split("code=")[1]
            logging.info("Coded:\n"+code)
            return code
        finally:
            driver.close()

    def get_token(self, grant_type):
        import urllib

        if grant_type == "authorization":
            code = self._signin_code()
            if code is not None:
                code = urllib.parse.unquote(code)
                logging.info("Decoded:\n"+code)
                params = {
                    "grant_type": "authorization_code",
                    "access_type": "offline",
                    "code": code,
                    "client_id": self.consumer_key,
                    "redirect_uri": "http://localhost",
                }
                headers = {"Content-Type": "application/x-www-form-urlencoded"}
                res = requests.post(
                    r"https://api.tdameritrade.com/v1/oauth2/token",
                    headers=headers,
                    data=params,
                )
                if res.ok:
                    res_body = res.json()
                    logging.info("Obtained access_token & refresh_token")
                    self.access_token = res_body["access_token"]
                    self.refresh_token = res_body["refresh_token"]
                else:
                    print(res)
                    print(res.json())
                    raise Exception(
                        f"API POST exception: Error {res.status_code}")
            else:
                raise Exception("Could not sign in and obtain code")
        elif grant_type == "refresh":
            params = {
                "grant_type": "refresh_token",
                "refresh_token": self.refresh_token,
                "client_id": self.consumer_key,
            }
            res = requests.post(
                r"https://api.tdameritrade.com/v1/oauth2/token",
                data=params,
            )
            if res.ok:
                res_body = res.json()
                self.access_token = res_body["access_token"]
                print(res_body["access_token"])
            else:
                print(res.json())

    def get_account_details(self):
        return requests.get(
            f"https://api.tdameritrade.com/v1/accounts/{os.environ['TDA_account_id']}",
            headers={"Authorization": f"Bearer {self.access_token}"},
        ).json()

    def get_account_details_with_positions(self):
        return requests.get(
            f"https://api.tdameritrade.com/v1/accounts/{os.environ['TDA_account_id']}?fields=positions",
            headers={"Authorization": f"Bearer {self.access_token}"},
        ).json()

    def get_quote(self, symbol):
        return requests.get(
            f"https://api.tdameritrade.com/v1/marketdata/{symbol}/quotes",
            params={"apikey": self.consumer_key},
        ).json()

    def calculate_commission(self):
        return 0

    def execute_order(self, event: OrderEvent) -> bool:
        if event is None:
            return False
        if event.type == "ORDER" and all(gk.check_gk(event, self.port.current_holdings) for gk in self.gatekeepers):
            for gk in self.gatekeepers:
                event = gk._alter_order(event, self.port.current_holdings)
            if event is not None:
                data = {
                    "orderType": "MARKET" if event.order_type == OrderType.MARKET else "LIMIT",
                    "session": "NORMAL",
                    "duration": "DAY",
                    "orderStrategyType": "SINGLE",
                    "orderLegCollection": [{
                        "instruction": "BUY" if event.direction == OrderPosition.BUY else "SELL",
                        "quantity": event.quantity,
                        "instrument": {
                            "symbol": event.symbol,
                            "assetType": "EQUITY"
                        }
                    }]
                }
                if data["orderType"] == "LIMIT":
                    data["price"] = event.signal_price
                res = requests.post(
                    f"https://api.tdameritrade.com/v1/accounts/{self.account_id}/orders",
                    data=json.dumps(data),
                    headers={
                        "Authorization": f"Bearer {self.access_token}",
                        "Content-Type": "application/json"
                    }
                )
                if not res.ok:
                    log_message(
                        f"Place Order Unsuccessful: {event.order_details()}\n{res.status_code}\n{res.json()}")
                    log_message(res.text)
                    return False
                self.pending_orders.append(event)
                self.check_filled_order()
                return True
        return False

    def check_filled_order(self):
        res = requests.get(
            f"https://api.tdameritrade.com/v1/accounts/{self.account_id}/orders",
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
        )
        prev_orders = res.json()
        print(prev_orders[-1])  # TODO: remove me
        for order in prev_orders:
            if order['quantity'] == order['filledQuantity'] and \
                    order['instrument']['symbol'] == self.pending_orders[0] and \
                    order['quantity'] == self.pending_orders[0].quantity:
                # Order filled
                event = self.pending_orders[0]
                event.trade_price = order['price']
                fill_event = FillEvent(event, self.calculate_commission())
                self.events.put(fill_event)
                self.pending_orders.pop(0)

    def cancel_order(self, order_id) -> bool:
        # NOTE: Unused and only skeleton.
        # TODO: Implement while improving TDABroker class
        res = requests.delete(
            f"https://api.tdameritrade.com/v1/accounts/{self.account_id}/orders/{order_id}",
            headers={"Authorization": f"Bearer {self.access_token}"}
        )
        if res.ok:
            return True
        return False

    def _filter_execute_order(self, event: OrderEvent) -> bool:
        return True

    def get_past_transactions(self):
        return requests.get(
            f"https://api.tdameritrade.com/v1/accounts/{self.account_id}/transactions",
            params={
                "type": "ALL",
            },
            headers={"Authorization": f"Bearer {self.access_token}"}
        ).json()

    def update_portfolio_positions(self, port: Portfolio):
        acct_details = self.get_account_details_with_positions()
        assert "securitiesAccount" in acct_details.keys() and "positions" in acct_details["securitiesAccount"].keys(
        ), "positions does not exit in queries acct details"
        position_list = acct_details["securitiesAccount"]["positions"]
        cur_holdings = dict()
        for pos_details in position_list:
            cur_holdings[pos_details["instrument"]["symbol"]] = dict(
                quantity=pos_details["longQuantity"],
                average_trade_price=pos_details["averagePrice"]
            )
        cur_holdings["cash"] = acct_details["securitiesAccount"]["currentBalances"]["cashBalance"]
        port.current_holdings.update(cur_holdings)
        port.write_curr_holdings()


class AlpacaBroker(Broker):
    def __init__(self, port, event_queue, gatekeepers: List[GateKeeper] = None):
        self.port: NaivePortfolio = port
        self.events = event_queue
        self.base_url = "https://paper-api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets/v2"
        self.api = alpaca_trade_api.REST(
            os.environ["alpaca_key_id"],
            os.environ["alpaca_secret_key"],
            self.base_url,
            api_version="v2",
        )
        if gatekeepers is None or len(gatekeepers) == 0:
            self.gatekeepers = [DummyGateKeeper()]
        else:
            self.gatekeepers = gatekeepers

    def _alpaca_endpoint(self, url, args: str):
        return requests.get(
            url + args,
            headers={
                "APCA-API-KEY-ID": os.environ["alpaca_key_id"],
                "APCA-API-SECRET-KEY": os.environ["alpaca_secret_key"],
            }
        ).json()

    """ ORDERS """

    def get_current_orders(self):
        return self.api.list_orders()

    def _filter_execute_order(self, event: OrderEvent) -> bool:
        return True

    def execute_order(self, event: OrderEvent) -> bool:
        if event is None:
            return False
        side = "buy" if event.direction == OrderPosition.BUY else "sell"
        if event.type == "ORDER" and all(gk.check_gk(event, self.port.current_holdings) for gk in self.gatekeepers):
            for gk in self.gatekeepers:
                event = gk._alter_order(event, self.port.current_holdings)
            if event is not None:
                try:
                    if event.order_type == OrderType.LIMIT:
                        order = self.api.submit_order(
                            symbol=event.symbol,
                            qty=event.quantity,
                            side=side,
                            type="limit",
                            time_in_force="day",
                            limit_price=event.signal_price,
                        )
                        event.trade_price = event.signal_price
                    else:
                        # todo: figure out a way to get trade_price for market orders
                        order = self.api.submit_order(
                            symbol=event.symbol,
                            qty=event.quantity,
                            side=side,
                            type="market",
                            time_in_force="day",
                        )
                        event.trade_price = event.signal_price
                except alpaca_trade_api.rest.APIError as e:
                    log_message(f"{self.api.get_account()}")
                    log_message(
                        f"Status Code [{e.status_code}] {e.code}: {str(e)}\nResponse: {e.response}")
                    return False
                if order.status == "accepted":
                    log_message(f"Order filled: {order}")
                    fill_event = FillEvent(event, self.calculate_commission())
                    self.events.put(fill_event)
                    return True
        return False

    def calculate_commission(self):
        return 0

    """ PORTFOLIO RELATED """

    def get_account_details(self):
        return self.api.get_account()

    def get_positions(self):
        return self.api.list_positions()

    def get_historical_bars(
        self, ticker, timeframe, start, end, limit: int = None
    ) -> pd.DataFrame:
        assert timeframe in ["1Min", "5Min", "15Min", "day", "1D"]
        if limit is not None:
            return self.api.get_barset(
                ticker, timeframe, start=start, end=end, limit=limit
            ).df
        return self.api.get_barset(ticker, timeframe, start=start, end=end).df

    def get_quote(self, ticker):
        return self.api.get_last_quote(ticker)


    def update_portfolio_positions(self, port: Portfolio):
        cur_holdings = dict()
        for pos_details in self.get_positions():
            cur_holdings[pos_details.symbol] = dict(
                quantity=float(pos_details.qty),
                average_trade_price=float(pos_details.avg_entry_price)
            )
        cur_holdings["cash"] = float(self.get_account_details().cash)
        port.current_holdings.update(cur_holdings)
    