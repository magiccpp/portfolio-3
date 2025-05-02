import os
import yfinance as yf
import pandas as pd
from git import Repo


text_file_path = '../training_data/'
git_path = '../price_data/'
stock_text_files = ['cac_40.txt', 'dax_40.txt', 'ftse_100.txt', 'sp_500.txt', 'etf.txt', 'omx_30.txt']
stock_ids = []

for text_file in stock_text_files:
  path_name = os.path.join(text_file_path, text_file)
  with open(path_name, "r") as f:
    stock_ids += [line.replace(' ', '-').strip() for line in f.readlines()]
    
    

multi = yf.Tickers(stock_ids)
hist  = multi.history(start="2023-01-01", auto_adjust=False, timeout=40)


def last_line_equals(df1: pd.DataFrame,
                     df2: pd.DataFrame,
                     date_col: str = None,
                     decimals: int = 4) -> bool:
    """
    Compare the last rows of df1 and df2:
      - The first column (or the one named in date_col) is compared as a date.
      - All other numeric columns are rounded to `decimals` places before comparing.
      - Returns False if either DataFrame is empty or columns differ.

    Args:
      df1, df2     : pandas DataFrames to compare
      date_col     : name of the date column; if None, uses the first column in df1
      decimals     : number of decimal places to round numeric columns to

    Returns:
      True if last rows match under the above rules, False otherwise.
    """
    # 1) quick checks
    if df1.empty or df2.empty:
        return False

    if not df1.columns.equals(df2.columns):
        return False

    # 2) decide which column is the "date" column
    if date_col is None:
        date_col = df1.columns[0]
    if date_col not in df1.columns:
        raise KeyError(f"date_col='{date_col}' not in DataFrame columns")

    # 3) grab last rows
    try:
        last1 = df1.iloc[-1]
        last2 = df2.iloc[-1]
    except IndexError:
        # one of them had zero rows
        return False

    # 4) compare the date column exactly (timestamps or strings)
    #    We force both sides through pd.to_datetime so e.g. "2020-01-01" == Timestamp("2020-01-01")
    dt1 = pd.to_datetime(last1[date_col])
    dt2 = pd.to_datetime(last2[date_col])
    if dt1 != dt2:
        return False

    # 5) find all other columns of numeric dtype and compare after rounding
    #    (you can adjust dtypes if you know they are all float, or use select_dtypes).
    other_cols = [c for c in df1.columns if c != date_col]

    for c in other_cols:
        v1, v2 = last1[c], last2[c]

        # handle NaNs: require both NaN to be considered equal
        if pd.isna(v1) and pd.isna(v2):
            continue
        if pd.isna(v1) or pd.isna(v2):
            return False

        # now round both sides to `decimals` places and compare
        # (works for float or int)
        if round(float(v1), decimals) != round(float(v2), decimals):
            return False

    # if we get here, date matched and all rounded numerics matched
    return True

def remove_nan_rows(df):
  """
  Removes rows from a pandas DataFrame where all columns are NaN.

  Args:
    df: The input pandas DataFrame.

  Returns:
    A new pandas DataFrame with the NaN-only rows removed.
  """
  return df.dropna(how='all')


updated_stocks = []
for stock in stock_ids:
  df = hist.xs(stock, axis=1, level='Ticker')
  df = remove_nan_rows(df)
  df.to_csv('/tmp/temp.csv', index=True)
  df = pd.read_csv('/tmp/temp.csv')
  save_path = os.path.join(git_path, stock + '.csv')
  # if the file exists in git_path, read it
  if os.path.exists(save_path):
    df_saved = pd.read_csv(git_path + stock + '.csv')
    if len(df) < len(df_saved):
      print(f"Warning: {stock} has fewer rows than the saved version. Skipping upload.")
      continue

    if last_line_equals(df, df_saved):
      print(f"Warning: {stock} is the same as the saved version. Skipping upload.")
      continue

  # save df to csv
  if len(df) == 0:
    print(f"Warning: {stock} has no data. Skipping upload.")
    continue

  df.to_csv(save_path, index=False)
  updated_stocks.append(save_path)
  print(f"Saved {stock} to {git_path + stock + '.csv'}")
  
  
BRANCH = "main"
repo = Repo(git_path)
repo.git.add("-A")
repo.index.commit(f"Add new price data")
origin = repo.remote(name="origin")
origin.push(refspec=f"{BRANCH}:{BRANCH}")