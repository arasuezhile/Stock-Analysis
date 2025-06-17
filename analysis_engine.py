# analysis_engine.py

import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns

# We import our DataHandler to make this script testable on its own
from data_handler import DataHandler
from datetime import datetime, timedelta

class AnalysisEngine:
    """Performs technical analysis including swing points, trendlines, and Fibonacci levels."""
    def __init__(self, data_df: pd.DataFrame):
        if data_df.empty: raise ValueError("Input DataFrame cannot be empty.")
        self.data = data_df.copy()
        self.trendlines = []
        self.fib_levels = []

    def find_swing_points(self, order: int = 5):
        """Identifies swing highs and swing lows."""
        high_peaks_indices, _ = find_peaks(self.data['High'], distance=order)
        low_peaks_indices, _ = find_peaks(-self.data['Low'], distance=order)
        self.data['swing_high'] = False
        self.data['swing_low'] = False
        self.data.iloc[high_peaks_indices, self.data.columns.get_loc('swing_high')] = True
        self.data.iloc[low_peaks_indices, self.data.columns.get_loc('swing_low')] = True
        print(f"Identified {self.data['swing_high'].sum()} swing highs and {self.data['swing_low'].sum()} swing lows.")

    def find_trendlines(self, tolerance_pct: float = 0.02, min_touches: int = 3):
        """Identifies and filters for the top 1 support and top 1 resistance trendlines."""
        swing_lows = self.data[self.data['swing_low']]
        swing_highs = self.data[self.data['swing_high']]
        
        def detect_lines(points, line_type):
            lines = []
            if len(points) < 2: return []
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    p1, p2 = points.iloc[i], points.iloc[j]
                    x1, y1 = (p1.name - self.data.index[0]).days, p1.Low if line_type == 'support' else p1.High
                    x2, y2 = (p2.name - self.data.index[0]).days, p2.Low if line_type == 'support' else p2.High
                    
                    if x2 == x1: continue
                    
                    # --- FIX IS HERE ---
                    # OLD problematic line: m, c = (y2 - y1) / (x2 - x1), y1 - m * x1
                    # NEW corrected lines:
                    m = (y2 - y1) / (x2 - x1) # Calculate slope first
                    c = y1 - m * x1          # Then calculate the intercept
                    # --- END OF FIX ---
                    
                    touches = sum(1 for k in range(len(points)) if abs(m * ((points.iloc[k].name - self.data.index[0]).days) + c - (points.iloc[k].Low if line_type == 'support' else points.iloc[k].High)) <= (points.iloc[k].Low if line_type == 'support' else points.iloc[k].High) * tolerance_pct)
                    if touches >= min_touches: lines.append({'type': line_type, 'slope': m, 'intercept': c, 'touches': touches})
            return lines

        support_lines, resistance_lines = detect_lines(swing_lows, 'support'), detect_lines(swing_highs, 'resistance')
        top_lines = []
        if support_lines: top_lines.append(sorted(support_lines, key=lambda x: x['touches'], reverse=True)[0])
        if resistance_lines: top_lines.append(sorted(resistance_lines, key=lambda x: x['touches'], reverse=True)[0])
        self.trendlines = top_lines
        print(f"Selected {len(self.trendlines)} top trendlines.")
        
    def calculate_fibonacci_levels(self):
        """Calculates Fibonacci retracement levels for the entire period's price range."""
        print("Calculating Fibonacci Retracement levels...")
        min_price = self.data['Low'].min()
        max_price = self.data['High'].max()
        price_range = max_price - min_price
        
        ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.fib_levels = [{'level': f'{ratio*100:.1f}%', 'price': max_price - ratio * price_range} for ratio in ratios]
        self.fib_levels.append({'level': '0.0% (High)', 'price': max_price})
        self.fib_levels.append({'level': '100.0% (Low)', 'price': min_price})
        print(f"Calculated {len(self.fib_levels)} Fibonacci levels.")

    def plot_analysis(self, stock_ticker: str, time_frame_title: str, show_fib: bool = False):
        """Creates a plot to visually verify all analysis."""
        sns.set_style("darkgrid")
        plt.figure(figsize=(18, 9))
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='skyblue', linewidth=1.5)
        plt.scatter(self.data[self.data['swing_high']].index, self.data[self.data['swing_high']]['High'], marker='v', color='red', s=70, zorder=5)
        plt.scatter(self.data[self.data['swing_low']].index, self.data[self.data['swing_low']]['Low'], marker='^', color='green', s=70, zorder=5)

        for line in self.trendlines:
            x_series = (self.data.index - self.data.index[0]).days
            y_series = line['slope'] * x_series + line['intercept']
            color = 'blue' if line['type'] == 'support' else 'purple'
            plt.plot(self.data.index, y_series, color=color, linestyle='--', linewidth=2, label=f"Top {line['type'].capitalize()} ({line['touches']} touches)")

        if show_fib and self.fib_levels:
            fib_colors = ['orange', 'gold', 'firebrick', 'darkcyan', 'magenta', 'black', 'black']
            for i, level in enumerate(sorted(self.fib_levels, key=lambda x: x['price'])):
                plt.axhline(y=level['price'], color=fib_colors[i], linestyle=':', linewidth=1.5, label=f"Fib {level['level']}")

        plt.title(f'{time_frame_title} Analysis for {stock_ticker}', fontsize=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.legend(fontsize=10, loc='upper left')
        plt.tight_layout()
        print(f"\nDisplaying {time_frame_title} plot...")
        plt.show()

# This block is for testing the script directly
if __name__ == '__main__':
    print("\n--- Running Multi-Timeframe Test Script with Fibonacci ---")
    stock_ticker = 'RELIANCE.NS'
    today = datetime.now()
    five_years_ago = today - timedelta(days=5*365)
    one_year_ago = today - timedelta(days=1*365)

    print(f"1. Fetching 5 years of data for {stock_ticker}...")
    handler = DataHandler(ticker=stock_ticker)
    long_term_data = handler.fetch_historical_data(start_date=five_years_ago.strftime('%Y-%m-%d'))

    if not long_term_data.empty:
        # --- LONG-TERM ANALYSIS ---
        print("\n" + "="*20 + " LONG-TERM ANALYSIS (5 Years) " + "="*20)
        analyzer_long_term = AnalysisEngine(data_df=long_term_data)
        analyzer_long_term.find_swing_points(order=50)
        analyzer_long_term.find_trendlines(tolerance_pct=0.03, min_touches=3)
        analyzer_long_term.calculate_fibonacci_levels()
        analyzer_long_term.plot_analysis(stock_ticker=stock_ticker, time_frame_title="Long-Term (5 Year)", show_fib=True)

        # --- SHORT-TERM ANALYSIS ---
        print("\n" + "="*20 + " SHORT-TERM ANALYSIS (1 Year) " + "="*20)
        short_term_data = long_term_data.loc[one_year_ago.strftime('%Y-%m-%d'):]
        analyzer_short_term = AnalysisEngine(data_df=short_term_data)
        analyzer_short_term.find_swing_points(order=15)
        analyzer_short_term.find_trendlines(tolerance_pct=0.015, min_touches=3)
        analyzer_short_term.plot_analysis(stock_ticker=stock_ticker, time_frame_title="Short-Term (1 Year)", show_fib=False)
        
        print("\n--- Test finished ---")
    else:
        print("Could not fetch data, aborting test.")
