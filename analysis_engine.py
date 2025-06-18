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
        self.elliott_waves, self.abc_projection, self.current_wave_status = [], {}, "No clear impulse wave pattern identified."

    def find_swing_points(self, order: int = 5):
        """Identifies swing highs and lows in the price data."""
        high_peaks_indices, _ = find_peaks(self.data['High'], distance=order)
        low_peaks_indices, _ = find_peaks(-self.data['Low'], distance=order)
        self.data['swing_high'], self.data['swing_low'] = False, False
        self.data.iloc[high_peaks_indices, self.data.columns.get_loc('swing_high')] = True
        self.data.iloc[low_peaks_indices, self.data.columns.get_loc('swing_low')] = True
        print(f"Identified {self.data['swing_high'].sum()} swing highs and {self.data['swing_low'].sum()} swing lows.")

    def find_elliott_wave_impulse(self, buffer: float = 0.03):
        """
        Identifies potential 5-wave impulse patterns using flexible rules.

        Args:
            buffer (float): The percentage tolerance for rule checking (e.g., 0.03 for 3%).
        """
        print(f"Searching for Elliott Wave patterns with a {buffer*100}% tolerance...")
        swings = self.data[(self.data['swing_high']) | (self.data['swing_low'])].copy()
        swings['price'] = np.where(swings['swing_low'], swings['Low'], swings['High'])
        
        valid_patterns = []
        for i in range(len(swings) - 5, -1, -1):
            p = swings.iloc[i:i+6]
            
            # --- CHANGE ---
            # This guard clause prevents errors on incomplete slices near the end of the data.
            if len(p) < 6:
                continue
            # --- END CHANGE ---
            
            if not (p.iloc[0]['swing_low'] and p.iloc[1]['swing_high'] and p.iloc[2]['swing_low'] and p.iloc[3]['swing_high'] and p.iloc[4]['swing_low'] and p.iloc[5]['swing_high']): continue
            
            p0, p1, p2, p3, p4, p5 = p['price'].values
            
            # --- RULE CHECKING WITH BUFFERS ---
            if p2 < p0: continue # Rule 1: No buffer
            if p4 < (p1 * (1 - buffer)): continue # Rule 3: Flexible overlap
            
            wave1_len, wave3_len, wave5_len = p1 - p0, p3 - p2, p5 - p4
            if wave3_len < (wave1_len * (1 - buffer)) or wave3_len < (wave5_len * (1 - buffer)): continue # Rule 2: Flexible length
            
            score = wave3_len / wave1_len
            valid_patterns.append({'pattern': p, 'score': score})

        if valid_patterns:
            best_pattern = sorted(valid_patterns, key=lambda x: x['score'], reverse=True)[0]
            p = best_pattern['pattern']
            self.elliott_waves.append({'type': 'impulse_up', 'points': p})

    def project_abc_correction(self, impulse_points):
        # This method is not used by the labeler but is kept for completeness
        pass

    def assess_current_wave_status(self):
        # This method is not used by the labeler but is kept for completeness
        pass

    def plot_analysis(self, stock_ticker: str, time_frame_title: str):
        # This method is not used by the labeler but is kept for completeness
        pass
