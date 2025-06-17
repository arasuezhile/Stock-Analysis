import pandas as pd
import sys

print(f"Python Executable: {sys.executable}")
print(f"Pandas Version: {pd.__version__}")

# Create a simple DatetimeIndex
dates = pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-05'])
dt_index = pd.DatetimeIndex(dates)

# Target a date not exactly in the index, using ffill
target_date = pd.Timestamp('2024-01-02')

try:
    loc = dt_index.get_loc(target_date, method='ffill')
    print(f"get_loc with ffill worked! Location: {loc}")
except TypeError as e:
    print(f"Minimal test failed with TypeError: {e}")
except Exception as e:
    print(f"Minimal test failed with other error: {e}")

# Try a date that is directly in the index (should always work)
target_date_exact = pd.Timestamp('2024-01-03')
try:
    loc_exact = dt_index.get_loc(target_date_exact)
    print(f"get_loc without method worked! Location: {loc_exact}")
except Exception as e:
    print(f"Minimal test (exact) failed with error: {e}")
