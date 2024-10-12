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
    def __init__(self, port, gatekeepers):
        self.port = port
        if gatekeepers is None or len(gatekeepers) == 0:
            self.gatekeepers = [DummyGateKeeper()]
        else:
            self.gatekeepers = gatekeepers

    @abstractmethod
    def _filter_execute_order(self, event: OrderEvent) -> bool:
        """
        Takes an OrderEvent and checks if it can be executed
        """
        raise NotImplementedError("Should implement _filter_execute_order()")

    @abstractmethod
    def execute_order(self, event: OrderEvent, event_queue, order_queue) -> bool:
        """
        Takes an OrderEvent and execute it, producing
        FillEvent that gets places into Events Queue
        """
        raise NotImplementedError("Should implement execute_order()")

    @abstractmethod
    def calculate_commission(
        self,
    ):
        """
        Takes an OrderEvent and calculates commission based
        on the details of the order it, add as argument for
        FillEvent
        """
        raise NotImplementedError("Implement calculate_commission()")

    def check_filled_order(self, event_queue):
        """More for live trading, since we send q request with no response from web api, we need to check via API for pending & filled orders"""
        pass

    def update_portfolio_positions(self):
        """Update for alpaca and TDA, use API call to update Portfolio class positions"""

    def get_account_details(self):
        """For live brokers"""
        pass

    def check_gk(self, event: OrderEvent):
        return all(gk.check_gk(event, self.port.current_holdings) for gk in self.gatekeepers)


"""
TODO: 
- include slippage and market impact
- make use of the "current" market data value to obtain a realistic fill cost.
"""


class SimulatedBroker(Broker):
    def __init__(self, bars, port, gatekeepers: List[GateKeeper] = None):
        super().__init__(port, gatekeepers)
        self.bars = bars

    def calculate_commission(self, quantity=None, fill_cost=None) -> float:
        return 0.0

    def _order_expired(self, latest_ohlc, order_event: OrderEvent):
        # check for expiry
        return latest_ohlc["timestamp"] > order_event.expires

    def _filter_execute_order(self, latest_snapshot, order_event: OrderEvent) -> bool:
        if order_event.order_type == OrderType.LIMIT:
            """
            print("Price data:", price_data_dict)
            print("OrderEvent: ", {
                "Symbol": event.symbol,
                "Price": event.signal_price,
                "Direction": event.direction,
                "Expires": event.expires    # in milliseconds
            })
            """
            low_price = latest_snapshot["low"]
            high_price = latest_snapshot["high"]
            if (
                order_event.direction == OrderPosition.SELL and order_event.signal_price > high_price
            ) or (
                order_event.direction == OrderPosition.BUY and order_event.signal_price < low_price
            ):
                return False
            if order_event.direction == OrderPosition.BUY and order_event.signal_price > high_price:
                order_event.signal_price = high_price
            if order_event.direction == OrderPosition.SELL and order_event.signal_price < low_price:
                order_event.signal_price = low_price
        return True

    def _put_fill_event(self, order_event: OrderEvent, event_queue):
        order_event.trade_price = order_event.signal_price
        fill_event = FillEvent(order_event, self.calculate_commission())
        event_queue.append(fill_event)
        return True

    def execute_order(self, event: OrderEvent, event_queue, order_queue) -> bool:
        if event is None:
            return False
        if event.type == "ORDER" and self.check_gk(event):
            latest_snapshot = self.bars.get_latest_bar(event.symbol)
            if event.order_type == OrderType.LIMIT:
                if not self._order_expired(latest_snapshot, event):
                    if not event.processed:
                        event.processed = True
                        order_queue.put(event)
                        return False
                    if self._filter_execute_order(latest_snapshot, event):
                        event.date = latest_snapshot['timestamp']
                        return self._put_fill_event(event, event_queue)
                    order_queue.put(event)
            elif event.order_type == OrderType.MARKET:
                event.signal_price = latest_snapshot["close"]
                event.date = latest_snapshot['timestamp']
                return self._put_fill_event(event, event_queue)
            return False
        return False


class IBBroker(Broker, EWrapper, EClient):
    def __init__(self, events):
        EClient.__init__(self, self)
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
        if msg.typeName == "openOrder" and msg.orderId == self.order_id and not self.fill_dict.has_key(msg.orderId):
            self.create_fill_dict_entry(msg)
        # Handle Fills
        elif (
            msg.typeName == "orderStatus" and msg.status == "Filled" and self.fill_dict[msg.orderId]["filled"] == False
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

    def create_fill(self, msg, event_queue):
        fd = self.fill_dict[msg.orderId]
        symbol = fd["symbol"]
        exchange = fd["exchange"]
        filled = msg.filled
        direction = fd["direction"]
        fill_cost = msg.avgFillPrice

        fill = FillEvent(datetime.datetime.utcnow(), symbol, exchange, filled, direction, fill_cost)
        self.fill_dict[msg.orderId]["filled"] = True
        event_queue.appendleft(fill)

    def execute_order(self, event) -> bool:
        if event is None:
            return False
        self.update_portfolio_positions()  # get latest curr_holdings from broker
        if event.type == "ORDER" and all(gk.check_gk(event, self.port.current_holdings) for gk in self.gatekeepers):
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
            full_cost = max(full_cost, 0.013 * quantity)
        else:
            full_cost = max(full_cost, 0.008 * quantity)
        full_cost = min(full_cost, 0.005 * quantity * fill_cost)

        return full_cost
