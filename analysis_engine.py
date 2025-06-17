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
    """
    Performs technical analysis on historical stock data.
    This engine can identify swing points and filter for the most
    significant trendlines based on the number of touches.
    """
    def __init__(self, data_df: pd.DataFrame):
        """
        Initializes the AnalysisEngine with a pandas DataFrame.

        Args:
            data_df (pd.DataFrame): The DataFrame containing OHLC price data.
                                    It must have a DatetimeIndex.
        """
        if data_df.empty:
            raise ValueError("Input DataFrame cannot be empty.")
        # Work on a copy to prevent modifying the original DataFrame passed in
        self.data = data_df.copy()
        self.trendlines = []
        print("AnalysisEngine initialized.")

    def find_swing_points(self, order: int = 5):
        """
        Identifies swing highs and lows in the price data using scipy.signal.find_peaks.
        A swing point is a point that is higher/lower than its neighbors.

        Args:
            order (int): Defines the significance of a swing point. It's the minimum
                         number of periods on each side of the swing point. A higher
                         order finds fewer, more major swing points.
        """
        # Find peaks in the 'High' price series for swing highs
        high_peaks_indices, _ = find_peaks(self.data['High'], distance=order)
        # Find peaks in the *inverted* 'Low' price series for swing lows
        low_peaks_indices, _ = find_peaks(-self.data['Low'], distance=order)
        
        # Add boolean columns to the DataFrame to mark these points
        self.data['swing_high'] = False
        self.data['swing_low'] = False
        self.data.iloc[high_peaks_indices, self.data.columns.get_loc('swing_high')] = True
        self.data.iloc[low_peaks_indices, self.data.columns.get_loc('swing_low')] = True
        
        print(f"Identified {self.data['swing_high'].sum()} swing highs and {self.data['swing_low'].sum()} swing lows.")

    def find_trendlines(self, tolerance_pct: float = 0.02, min_touches: int = 3):
        """
        Identifies all potential trendlines and then filters for the single best
        support and resistance line based on the maximum number of touches.

        Args:
            tolerance_pct (float): The percentage tolerance. A point is considered to "touch" a line
                                   if its price is within this percentage of the line's value.
            min_touches (int): The minimum number of swing points a line must touch to be
                               considered a valid potential trendline.
        """
        print(f"Finding all potential trendlines with tolerance={tolerance_pct*100}%...")
        swing_lows = self.data[self.data['swing_low']]
        swing_highs = self.data[self.data['swing_high']]
        
        def detect_lines(points, line_type):
            """A helper function to detect all lines meeting the min_touches criteria."""
            lines = []
            if len(points) < 2:
                return []
            
            # Iterate through all possible pairs of points to form a line
            for i in range(len(points)):
                for j in range(i + 1, len(points)):
                    p1, p2 = points.iloc[i], points.iloc[j]
                    
                    # Get coordinates for line equation y = mx + c
                    # x is days from the start of the data, y is price
                    x1 = (p1.name - self.data.index[0]).days
                    y1 = p1.Low if line_type == 'support' else p1.High
                    x2 = (p2.name - self.data.index[0]).days
                    y2 = p2.Low if line_type == 'support' else p2.High
                    
                    if x2 == x1: continue # Skip vertical lines
                    m = (y2 - y1) / (x2 - x1) # Slope
                    c = y1 - m * x1          # Intercept
                    
                    # Count how many of the other swing points touch this line
                    touches = 0
                    for k in range(len(points)):
                        pk = points.iloc[k]
                        xk = (pk.name - self.data.index[0]).days
                        yk_actual = pk.Low if line_type == 'support' else pk.High
                        yk_line = m * xk + c
                        
                        # Check if the point's price is within the tolerance percentage of the line
                        if abs(yk_line - yk_actual) <= yk_actual * tolerance_pct:
                            touches += 1
                            
                    # If the line is significant enough, store it
                    if touches >= min_touches:
                        lines.append({'type': line_type, 'slope': m, 'intercept': c, 'touches': touches})
            return lines

        # Detect all potential support and resistance lines
        support_lines = detect_lines(swing_lows, 'support')
        resistance_lines = detect_lines(swing_highs, 'resistance')
        
        print(f"Found {len(support_lines)} potential support lines and {len(resistance_lines)} potential resistance lines.")
        print("Filtering for the best lines based on number of touches...")
        
        # --- RANKING AND FILTERING LOGIC ---
        top_lines = []
        if support_lines:
            # Sort support lines by the number of touches (most first) and pick the top one
            best_support = sorted(support_lines, key=lambda x: x['touches'], reverse=True)[0]
            top_lines.append(best_support)
            
        if resistance_lines:
            # Sort resistance lines by the number of touches and pick the top one
            best_resistance = sorted(resistance_lines, key=lambda x: x['touches'], reverse=True)[0]
            top_lines.append(best_resistance)
            
        self.trendlines = top_lines
        print(f"Selected {len(self.trendlines)} top trendlines.")

    def plot_analysis(self, stock_ticker: str, time_frame_title: str):
        """
        Creates a plot to visualize the price, swing points, and final trendlines.

        Args:
            stock_ticker (str): The ticker symbol for the plot title.
            time_frame_title (str): A title describing the timeframe (e.g., "Long-Term").
        """
        sns.set_style("darkgrid")
        plt.figure(figsize=(18, 9))

        # Plot the closing price line
        plt.plot(self.data.index, self.data['Close'], label='Close Price', color='skyblue', linewidth=1.5)
        # Add markers for the identified swing points
        plt.scatter(self.data[self.data['swing_high']].index, self.data[self.data['swing_high']]['High'], label='Swing High', marker='v', color='red', s=70, zorder=5)
        plt.scatter(self.data[self.data['swing_low']].index, self.data[self.data['swing_low']]['Low'], label='Swing Low', marker='^', color='green', s=70, zorder=5)

        # Draw the final, filtered trendlines on the chart
        for line in self.trendlines:
            x_series = (self.data.index - self.data.index[0]).days
            y_series = line['slope'] * x_series + line['intercept']
            color = 'blue' if line['type'] == 'support' else 'purple'
            label = f"Top {line['type'].capitalize()} ({line['touches']} touches)"
            plt.plot(self.data.index, y_series, color=color, linestyle='--', linewidth=2, label=label)

        # Final plot styling
        plt.title(f'{time_frame_title} Analysis for {stock_ticker}', fontsize=20)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Price', fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        print(f"\nDisplaying {time_frame_title} plot...")
        plt.show()

# This block executes only when you run `python3 analysis_engine.py` directly.
# It's used for testing the functionalities of this specific module.
if __name__ == '__main__':
    print("\n--- Running Multi-Timeframe Test Script for AnalysisEngine ---")
    stock_ticker = 'RELIANCE.NS'
    today = datetime.now()
    five_years_ago = today - timedelta(days=5*365)
    one_year_ago = today - timedelta(days=1*365)

    # 1. Fetch one long-term dataset using the DataHandler
    print(f"1. Fetching 5 years of data for {stock_ticker}...")
    handler = DataHandler(ticker=stock_ticker)
    long_term_data = handler.fetch_historical_data(start_date=five_years_ago.strftime('%Y-%m-%d'))

    if not long_term_data.empty:
        # --- Run the Long-Term Analysis ---
        print("\n" + "="*20 + " LONG-TERM ANALYSIS (5 Years) " + "="*20)
        analyzer_long_term = AnalysisEngine(data_df=long_term_data)
        analyzer_long_term.find_swing_points(order=50) # Higher order for major swings
        analyzer_long_term.find_trendlines(tolerance_pct=0.03, min_touches=3)
        analyzer_long_term.plot_analysis(stock_ticker=stock_ticker, time_frame_title="Long-Term (5 Year)")

        # --- Run the Short-Term Analysis ---
        print("\n" + "="*20 + " SHORT-TERM ANALYSIS (1 Year) " + "="*20)
        # Create the 1-year dataset by slicing the long-term data
        short_term_data = long_term_data.loc[one_year_ago.strftime('%Y-%m-%d'):]
        analyzer_short_term = AnalysisEngine(data_df=short_term_data)
        analyzer_short_term.find_swing_points(order=15) # Lower order for more recent swings
        analyzer_short_term.find_trendlines(tolerance_pct=0.015, min_touches=3)
        analyzer_short_term.plot_analysis(stock_ticker=stock_ticker, time_frame_title="Short-Term (1 Year)")
        
        print("\n--- Test finished ---")
    else:
        print("Could not fetch data, aborting test.")
