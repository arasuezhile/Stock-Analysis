# analysis_engine.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as patheffects
from data_handler import DataHandler

class AnalysisEngine:
    """Performs technical analysis including predictive Elliott Wave status assessment."""
    def __init__(self, data_df: pd.DataFrame):
        if data_df.empty: raise ValueError("Input DataFrame cannot be empty.")
        self.data = data_df.copy()
        self.elliott_waves, self.abc_projection, self.current_wave_status = [], {}, "No clear, recent impulse wave pattern identified."

    def find_swing_points(self, order: int = 5):
        """Identifies swing highs and lows in the price data."""
        high_peaks_indices, _ = find_peaks(self.data['High'], distance=order)
        low_peaks_indices, _ = find_peaks(-self.data['Low'], distance=order)
        self.data['swing_high'], self.data['swing_low'] = False, False
        self.data.iloc[high_peaks_indices, self.data.columns.get_loc('swing_high')] = True
        self.data.iloc[low_peaks_indices, self.data.columns.get_loc('swing_low')] = True
        print(f"Identified {self.data['swing_high'].sum()} swing highs and {self.data['swing_low'].sum()} swing lows.")

    def find_elliott_wave_impulse(self):
        """Identifies ALL potential 5-wave impulse patterns and selects the best one."""
        print("Searching for best quality Elliott Wave impulse pattern...")
        swings = self.data[(self.data['swing_high']) | (self.data['swing_low'])].copy()
        swings['price'] = np.where(swings['swing_low'], swings['Low'], swings['High'])
        
        valid_patterns = []
        for i in range(len(swings) - 5, -1, -1):
            p = swings.iloc[i:i+6]
            if not (p.iloc[0]['swing_low'] and p.iloc[1]['swing_high'] and p.iloc[2]['swing_low'] and p.iloc[3]['swing_high'] and p.iloc[4]['swing_low'] and p.iloc[5]['swing_high']): continue
            p0, p1, p2, p3, p4, p5 = p['price'].values
            if p2 < p0 or p4 < p1: continue
            wave1_len, wave3_len, wave5_len = p1 - p0, p3 - p2, p5 - p4
            if wave3_len < wave1_len or wave3_len < wave5_len: continue
            
            score = wave3_len / wave1_len
            valid_patterns.append({'pattern': p, 'score': score})

        if valid_patterns:
            best_pattern = sorted(valid_patterns, key=lambda x: x['score'], reverse=True)[0]
            p = best_pattern['pattern']
            self.elliott_waves.append({'type': 'impulse_up', 'points': p})
            print(f"Found best-fit 5-wave impulse pattern ending at {p.index[-1].date()}")
            self.project_abc_correction(p)
            self.assess_current_wave_status()
            
    def project_abc_correction(self, impulse_points):
        """Projects a 3-wave A-B-C correction based on a completed 5-wave impulse."""
        print("Projecting A-B-C corrective wave...")
        p4, p5 = impulse_points['price'].iloc[4], impulse_points['price'].iloc[5]
        wave_a_target_price, wave_a_magnitude = p4, p5 - p4
        wave_b_target_price = wave_a_target_price + (wave_a_magnitude * 0.5)
        wave_c_target_price = wave_b_target_price - wave_a_magnitude
        
        last_date = impulse_points.index[-1]
        time_delta_per_wave = timedelta(days=max(30, (last_date - impulse_points.index[0]).days / 5))
        
        date_a, date_b, date_c = last_date + time_delta_per_wave, last_date + 2*time_delta_per_wave, last_date + 3*time_delta_per_wave
        self.abc_projection = {'p5': (last_date, p5), 'A': (date_a, wave_a_target_price), 'B': (date_b, wave_b_target_price), 'C': (date_c, wave_c_target_price)}
        print(f"Projected Correction Targets: A={wave_a_target_price:.2f}, B={wave_b_target_price:.2f}, C={wave_c_target_price:.2f}")

    def assess_current_wave_status(self):
        """Assesses where the current price is within the identified wave structure."""
        if not self.abc_projection: return
        today = pd.Timestamp.now(tz=self.data.index.tz if self.data.index.tz is not None else 'UTC')
        proj = self.abc_projection
        
        if today > proj['C'][0]: self.current_wave_status = "A prior impulse and its correction appear complete. A new trend may be forming."
        elif today > proj['B'][0]: self.current_wave_status = f"The stock appears to be in Corrective **Wave C**. Projected target: **₹{proj['C'][1]:.2f}**"
        elif today > proj['A'][0]: self.current_wave_status = f"The stock appears to be in Corrective **Wave B**. Projected target: **₹{proj['B'][1]:.2f}**"
        elif today > proj['p5'][0]: self.current_wave_status = f"The stock appears to be in Corrective **Wave A**. Projected target: **₹{proj['A'][1]:.2f}**"

    def plot_analysis(self, stock_ticker: str, time_frame_title: str):
        """Creates a plot to visually verify all analysis."""
        sns.set_style("darkgrid")
        fig, ax = plt.subplots(figsize=(18, 9))
        ax.plot(self.data.index, self.data['Close'], label='Close Price', color='skyblue', linewidth=1.5, zorder=1)
        
        if not self.elliott_waves:
            print("No valid Elliott Wave pattern found to plot.")
        else:
            for wave in self.elliott_waves:
                points = wave['points']
                ax.plot(points.index, points['price'], color='#FFDE03', marker='o', linestyle='-', linewidth=4, markersize=10, zorder=3, label='Elliott Wave (0-5)')
                labels = ['0', '1', '2', '3', '4', '5']
                for i, (idx, row) in enumerate(points.iterrows()):
                    ax.text(idx, row['price'], labels[i], fontsize=16, fontweight='bold', ha='center', va='bottom', zorder=4, path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])

            if self.abc_projection:
                proj_points = self.abc_projection
                p5_date, p5_price = proj_points['p5']
                dates = [p5_date] + [proj_points[p][0] for p in ['A', 'B', 'C']]
                prices = [p5_price] + [proj_points[p][1] for p in ['A', 'B', 'C']]
                ax.plot(dates, prices, color='red', linestyle='--', linewidth=3, marker='o', markersize=10, zorder=3, label='Projected Correction (A-B-C)')
                labels = ['A', 'B', 'C']
                for i, label in enumerate(labels):
                    ax.text(dates[i+1], prices[i+1], label, fontsize=18, fontweight='bold', color='red', ha='center', va='bottom', zorder=4, path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
                
                print("Dynamically zooming chart to relevant area...")
                buffer = timedelta(days=30) 
                zoom_start_date, zoom_end_date = points.index[0] - buffer, dates[-1] + buffer
                ax.set_xlim([zoom_start_date, zoom_end_date])
                
        ax.set_title(f'{time_frame_title} Analysis for {stock_ticker}', fontsize=20)
        ax.legend(fontsize=12, loc='upper left')
        plt.tight_layout()
        print(f"\nDisplaying {time_frame_title} plot...")
        plt.show()

def run_full_analysis(stock_ticker: str):
    """
    Main function to run the complete multi-timeframe analysis for a stock.
    """
    today = datetime.now()
    handler = DataHandler(ticker=stock_ticker)

    # --- 1. LONG-TERM DAILY ANALYSIS ---
    print("\n" + "="*25 + " 1. LONG-TERM DAILY ANALYSIS " + "="*25)
    long_term_start = (today - timedelta(days=5*365)).strftime('%Y-%m-%d')
    long_term_data = handler.fetch_historical_data(start_date=long_term_start, interval='1d')
    if not long_term_data.empty:
        analyzer_lt = AnalysisEngine(data_df=long_term_data)
        analyzer_lt.find_swing_points(order=50)
        analyzer_lt.find_elliott_wave_impulse()
        analyzer_lt.plot_analysis(stock_ticker=stock_ticker, time_frame_title="Long-Term (Daily) Wave Analysis")
        print("\n" + "="*25 + " LONG-TERM WAVE STATUS " + "="*25)
        print(analyzer_lt.current_wave_status)
        print("="*75)
    else:
        print("Could not perform Long-Term Analysis.")

    # --- 2. SHORT-TERM HOURLY ANALYSIS ---
    print("\n" + "="*25 + " 2. SHORT-TERM HOURLY ANALYSIS " + "="*25)
    short_term_start = (today - timedelta(days=720)).strftime('%Y-%m-%d')
    short_term_data = handler.fetch_historical_data(start_date=short_term_start, interval='1h')
    if not short_term_data.empty:
        analyzer_st = AnalysisEngine(data_df=short_term_data)
        analyzer_st.find_swing_points(order=30)
        analyzer_st.find_elliott_wave_impulse()
        analyzer_st.plot_analysis(stock_ticker=stock_ticker, time_frame_title="Short-Term (Hourly) Wave Analysis & Projection")
        print("\n" + "="*25 + " SHORT-TERM WAVE STATUS " + "="*25)
        print(analyzer_st.current_wave_status)
        print("="*75)
    else:
        print("Could not perform Short-Term Analysis.")

    print("\n--- Full Analysis Finished ---")

# This block allows you to run the full analysis from the command line
if __name__ == '__main__':
    # You can change the ticker here to analyze a different stock
    run_full_analysis(stock_ticker='RELIANCE.NS')
