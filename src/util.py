import atexit
import json
import sys
from sklearn.linear_model import LinearRegression

import pandas as pd
from datetime import timedelta
import yfinance as yfin
from pandas_datareader import data as pdr
from safeRegressors import SafeRandomForestRegressor, SafeSVR

import os
import time
import numpy as np
import pickle
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import logging



logger = logging.getLogger('inference')
logger.setLevel(logging.DEBUG)  # Set the logging level


NUMBER_RECENT_SECONDS = 72000
MIN_TOTAL_DATA_PER_STOCK = 1000
MIN_TRAINING_DATA_PER_STOCK = 500
MIN_TEST_DATA_PER_STOCK = 300
TIMEOUT = 120

INTEREST_RATE = 0.0497    # Current interest rate accessible for USD
ANNUAL_TRADING_DAYS = 252
MAX_RISK = 0.08

def get_tickers(ticker_list_file):
  with open(f'data/{ticker_list_file}', 'r') as f:
    tickers = f.readlines()
    # replace space to dash '-'
    tickers = [ticker.replace(' ', '-').strip() for ticker in tickers]
    return tickers

def nth_weekday_of_month(year, month, index, weekday):
  """
  Find the nth occurrence of a specific weekday in a given month.

  :param year: The year as an integer.
  :param month: The month as an integer (1-12).
  :param index: The index of the occurrence (1st, 2nd, 3rd, etc.)
  :param weekday: The day of the week as an integer where Monday is 1 and Sunday is 7.
  :return: The date of the nth weekday.
  """

  weekday = weekday - 1
  # Start at the beginning of the month
  first_day_of_month = pd.Timestamp(year=year, month=month, day=1)
  # Find the first occurrence of the specific weekday
  first_weekday = first_day_of_month + timedelta(days=((weekday - first_day_of_month.weekday()) + 7) % 7)

  # Add (index - 1) weeks to the first occurrence of the weekday
  nth_weekday = first_weekday + timedelta(weeks=index-1)
  return nth_weekday.day

def is_file_downloaded_recently(file_path, seconds=NUMBER_RECENT_SECONDS):
  if not os.path.exists(file_path):
    return False
  file_age = time.time() - os.path.getmtime(file_path)
  return file_age <= seconds

def get_table_by_id_fred(id, path, n_features,
                         start='1950-01-01', end="2024-01-01", if_log=True):
  feature_columns = []
  if path is None:
    path = 'data/fred'

  file_path = os.path.join(path, f'{id}.csv')
  if not is_file_downloaded_recently(file_path):
    print(f'Metric: {id} need to be refreshed...')
    df = pdr.get_data_fred(id, start='1950-01-01', end=None)
    df.to_csv(f'{path}/{id}.csv')

  df = pd.read_csv(os.path.join(path, f'{id}.csv'), index_col='DATE', parse_dates=True)
  df = df[start:end]

  if if_log:
    df[f'log_{id}'] = np.log(df[id])

  n_days = [int(2**n) for n in range(n_features)]
  for n in n_days:
    if if_log:
      name = f'log_{id}_diff_{n}'
      df[name] = df[f'log_{id}'] - df[f'log_{id}'].shift(n)
    else:
      name = f'{id}_diff_{n}'
      df[name] = df[id] - df[id].shift(n)
    feature_columns.append(name)
  return df, feature_columns

def merge_fred(df, id, n_features, start, end, release_week_index, release_week_day, if_log=True):
  path = 'data/fred'
  df_new, columns = get_table_by_id_fred(id, path, n_features, start=start, end=end, if_log=if_log)


  def get_last_metric_date(row, release_week_index, release_week_day):
    year = row.name.year
    month = row.name.month
    day = row.name.day

    release_date = nth_weekday_of_month(year, month, release_week_index, release_week_day)
    if day <= release_date:
      if month == 1:
        year -= 1
        month = 11
      elif month == 2:
        year -= 1
        month = 12
      else:
        month -= 2
    else:
      if month == 1:
        year -= 1
        month = 12
      else:
        month -= 1

    return pd.to_datetime(f"{year}-{month}-01")

  df['LAST_METRIC_DATE'] = df.apply(get_last_metric_date, axis=1,
                                    args=(release_week_index, release_week_day))

  df = pd.merge_asof(df, df_new[columns], left_on='LAST_METRIC_DATE', right_index=True)
  # delete the column 'LAST_METRIC_DATE'
  df = df.drop(columns=['LAST_METRIC_DATE'])
  return df, columns

def remove_nan(df, type='top'):
  if type == 'top':
    for i in range(len(df)):
      if df.iloc[i].isnull().any() == False:
        break
    df_top = df[:i]
    df = df[i:]

    return df, df_top

  elif type == 'bottom':
    for i in range(1, len(df)):
      if df.iloc[-i].isnull().any() == False:
        break
    df_tail = df[-i:]
    df = df[:-i]
    return df, df_tail


def add_features(df, n_features):
  feature_columns = []
  for i in range(n_features):
    n_days = 2**i

    df[f'log_price_diff_{n_days}'] = np.log(df['Adj Close']/df['Adj Close'].shift(n_days))
    #df[f'price_diff_{n_days}'] = pd.to_numeric(df[f'price_diff_{n_days}'], errors='coerce')
    #log_volume = np.log(df['Volume']+1e-8)
    #df[f'log_volume_diff_{n_days}'] = log_volume - log_volume.shift(n_days)
    feature_columns.append(f'log_price_diff_{n_days}')
    #feature_columns.append(f'log_volume_diff_{n_days}')
    #feature_columns.append(f'volume_diff_{n_days}')
  return df, feature_columns


# Map the stock suffixes to their base currencies
currency_mapping = {
  '.ST': 'SEK',
  '.DE': 'EUR',
  '.L': 'GBP',
  '.SS': 'RMB'
}

# Map currency pairs to directions
conversion_mapping = {
  ('SEK', 'USD'): ('DEXSDUS', True, 'SEK=X'),
  ('EUR', 'USD'): ('DEXUSEU', False, 'EURUSD=X'),
  ('GBP', 'USD'): ('DEXUSUK', False, 'GBPUSD=X'),
  ('RMB', 'USD'): ('DEXCHUS', True, 'CNY=X')
}

def get_currency_pair(stock_suffix, base_currency):
    stock_base_currency = currency_mapping.get(stock_suffix, 'USD')
    if base_currency == stock_base_currency:
        return None, None, None  # No conversion needed
    else:
        return conversion_mapping.get((stock_base_currency, base_currency))


def read_and_filter_exchange_rates(exchange_name, exchange_name_yahoo, path='data/fred'):
  return read_and_filter(exchange_name, exchange_name_yahoo, path)

def read_and_filter(exchange_name, exchange_name_yahoo, path):
  filepath = f'{path}/{exchange_name}.csv'
  if not is_file_downloaded_recently(filepath):
    print(f'Metric: {exchange_name} need to be refreshed...')
    df = pdr.get_data_fred(exchange_name, start='1950-01-01', end=None)
    print('df:')
    print(df)
    print(df.columns)
    # fred data is out of date, we use yahoo data as complement
    df2 = yfin.download(exchange_name_yahoo, start='1950-01-01', end=None)
    print('df2:')
    print(df2)
    print(df2.columns)
    # remove the second index of df2
    if isinstance(df2.columns, pd.MultiIndex):
      df2.columns = df2.columns.get_level_values(0)
    index_name = df.index.name
    df = df.join(df2[['Close']], how='outer')
    df.index.name = index_name

    df[exchange_name].fillna(df['Close'], inplace=True)
    df.drop('Close', axis=1, inplace=True)
    df.dropna(inplace=True)
    df.to_csv(f'{path}/{exchange_name}.csv')

  df = pd.read_csv(filepath, index_col='DATE', parse_dates=True)
  return df

def convert(df, exchange_name, inversion, exchange_name_yahoo):
  df_rate = read_and_filter_exchange_rates(exchange_name, exchange_name_yahoo)
  start = max(df.index[0], df_rate.index[0])
  df = df[df.index >= start]
  df_rate = df_rate[df_rate.index >= start]

  df_rate = df_rate[[exchange_name]]
  if inversion:
    df_rate[exchange_name] = 1/df_rate[exchange_name]
  df_merged = pd.merge_asof(df, df_rate, left_index=True, right_index=True, direction='nearest')
  df_merged['Adj Close'] = df_merged['Adj Close'] * df_merged[exchange_name]
  return df_merged[['Adj Close', 'Volume']]


class FileLock:
    def __init__(self, lock_path, max_retries=10, delay_seconds=5):
        self.lock_path = lock_path
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.lock_acquired = False

    def __enter__(self):
        attempts = 0
        while attempts < self.max_retries:
            try:
                os.makedirs(self.lock_path)
                atexit.register(self.release)
                self.lock_acquired = True
                return self
            except FileExistsError:
                time.sleep(self.delay_seconds)
                attempts += 1
        raise RuntimeError(f"Could not acquire lock for {self.lock_path} after retries")

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    def release(self):
        if self.lock_acquired:
            try:
                os.rmdir(self.lock_path)
                self.lock_acquired = False
            except OSError:
                pass

def load_latest_price_data(stock_name, start='1950-01-01', end=None, save=True, force_download=False):
  file_path = f'data/prices/{stock_name}.csv'

  if end is not None:
    # get the number of seconds from end to now
    now = pd.Timestamp.now()
    seconds = (now - pd.to_datetime(end)).total_seconds()
  else:
    seconds = NUMBER_RECENT_SECONDS

  lock_path = f'data/prices/.lock_{stock_name}'

  with FileLock(lock_path):
    if not is_file_downloaded_recently(file_path, seconds=seconds) or force_download:
      print('Preparing downloading...', stock_name)
      #data = yfin.download(stock_name, start=start, end=None, auto_adjust=True, timeout=40)



      data = yfin.Ticker(stock_name).history(start=start, end=None, auto_adjust=False, timeout=40)
      data.index = data.index.date.astype(str)

      if len(data) > 100:
        if save:
          data.to_csv(file_path, index_label='Date')
      else:
        print(f'Cannot download {stock_name}, using old data...')

    df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
  # Release the lock before potential return

  if len(df) < 100:
    return None

  stock_suffix = '.' + stock_name.split('.')[-1]
  exchange_name, needs_inversion, exchange_name_yahoo = get_currency_pair(stock_suffix, 'USD')
  if exchange_name is not None:
    df = convert(df, exchange_name, needs_inversion, exchange_name_yahoo)
  return df


def   get_X_y_by_stock(stock_name, period, start, end, split_date, force_download):
  try:
    df = load_latest_price_data(stock_name, start=start, end=end, force_download=force_download)
  except FileNotFoundError:
    print(f'Cannot find data for: {stock_name}')
    return None, None, None, None

  print(f'processing {stock_name}...')
  if df is None or len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name}')
    return None, None, None, None

  if len(df) == 0:
    print(f'empty table...')
    return None, None, None, None

  df, feature_columns = add_features(df, 10)

  # the predict is the log return of period days.
  df['log_predict'] = np.log(df['Adj Close'].shift(-period) / df['Adj Close'])


  timestamp = df.index[0]
  earliest_date = timestamp.strftime('%Y-%m-%d')
  start = earliest_date
  end = None



  df, columns = merge_fred(df, 'M2SL', 6, start, end, 4, 2, if_log=True)
  feature_columns += columns


  df, columns = merge_fred(df, 'UNRATE', 6, start, end, 1, 5, if_log=False)
  feature_columns += columns

  df, columns = merge_fred(df, 'FEDFUNDS', 6, start, end, 1, 5, if_log=False)
  feature_columns += columns

  df, _ = remove_nan(df, type='top')
  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name} after removing nan from the top')
    return None, None, None, None
  df, _ = remove_nan(df, type='bottom')
  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name} after removing nan from the bottom')
    return None, None, None, None

  df = df[feature_columns + ['log_predict']]
  df.dropna(inplace=True)

  if len(df) < MIN_TOTAL_DATA_PER_STOCK:
    print(f'Cannot find enough data for: {stock_name}')
    return None, None, None, None

  df_test = df[df.index >= split_date]
  df_train = df[df.index < split_date]

  if len(df_train) < MIN_TRAINING_DATA_PER_STOCK:
    print(f'Cannot find enough training data for: {stock_name}')
    return None, None, None, None
  if len(df_test) < MIN_TEST_DATA_PER_STOCK:
    print(f'Cannot find enough test data for: {stock_name}')
    return None, None, None, None
  df_train_X = df_train[feature_columns]
  df_train_y = df_train[['log_predict']]
  df_test_X = df_test[feature_columns]
  df_test_y = df_test[['log_predict']]

  return df_train_X, df_train_y, df_test_X, df_test_y

def save_pkl(object, file):
  with open(file, 'wb') as f:
    pickle.dump(object, f)


def load_pkl(file):
  with open(file, 'rb') as f:
    return pickle.load(f)


class TruncationTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, k):
    self.k = k

  def fit(self, X, y=None):
    # Since this transformer is stateless, the fit method only needs to pass
    return self

  def transform(self, X, y=None):
    return X[:, :self.k]

def get_pipline_rf(params):
  pipeline = Pipeline([
          ('truncate', TruncationTransformer(k=params['k'])), # Adjust 'k' as needed
          ('regress', SafeRandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            min_samples_split=params['min_samples_split'],
            min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],
            bootstrap=params['bootstrap'],
            max_leaf_nodes=params['max_leaf_nodes'],
            timeout=TIMEOUT
      ))])

  return pipeline


def get_pipline_svr(params):
  pipeline = Pipeline([
          ('scaler', StandardScaler()),  # Add scaler here
          ('truncate', TruncationTransformer(k=params['k'])), # Adjust 'k' as needed
          ('regress', SafeSVR(
            C=params['C'],
            epsilon=params['epsilon'], kernel=params['kernel'],
            gamma=params['gamma'], timeout=TIMEOUT
      ))])

  return pipeline

def create_if_not_exist(path):
  if not os.path.exists(path):
    os.makedirs(path)
    print(f'Creating {path}...')


def portfolio_volatility_log_return(weights, covariance):
    return np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))

def portfolio_log_return(weights, returns):
    return np.sum(returns*weights)

def portfolio_volatility(weights, covariance_log_returns):
    covariance_returns = np.exp(covariance_log_returns) - 1
    return np.sqrt(np.dot(weights.T, np.dot(covariance_returns, weights)))

def portfolio_return(weights, log_returns):
    returns = np.exp(log_returns) - 1
    return np.sum(returns*weights)

def generate_features(data_dir, df_train_X_all, df_train_y_all, valid_tickers):
  feature_names = list(df_train_X_all[0].columns)
  coefficients = np.zeros(len(feature_names))
  for i in range(len(valid_tickers)):
      ticker = valid_tickers[i]
      df_train_X_stock = df_train_X_all[i]
      df_train_y_stock = df_train_y_all[i]
      # concat the gspc data to the d_train_X_stock

      # scale the features
      scaler = StandardScaler()
      df_train_X_stock = scaler.fit_transform(df_train_X_stock)

      # Apply feature selection using SelectKBest
      #selector = SelectKBest(f_regression, k=5)  # Select 5 best features
      # X_new = selector.fit_transform(df_train_X_stock, df_train_y_stock['log_predict'])

      # Fit linear regression model
      model = LinearRegression()
      model.fit(df_train_X_stock, df_train_y_stock['log_predict'])

      # Get the coefficients of the selected features:
      coefficients += model.coef_

      # Mapping back to feature names (assuming you have a list of feature names)

  sorted_scores = np.abs(coefficients).argsort()[::-1]  # Sort indices in descending order of scores
  feature_names = np.array(feature_names)
  sorted_features = feature_names[sorted_scores]

  with open(f'{data_dir}/sorted_features.txt', 'w') as f:
    for item in sorted_features:
      f.write("%s\n" % item)
  return sorted_features


def min_func_sharpe(weights, returns, covariance, risk_free_rate):
    portfolio_ret = portfolio_log_return(weights, returns)
    portfolio_vol = portfolio_volatility_log_return(weights, covariance)
    sharpe_ratio = (portfolio_ret - risk_free_rate) / portfolio_vol
    return -sharpe_ratio # Negate Sharpe ratio because we minimize the function

def min_func_two_sigma(weights, returns, covariance, risk_free_rate):
    portfolio_ret = portfolio_log_return(weights, returns)
    portfolio_vol = portfolio_volatility_log_return(weights, covariance)
    value = portfolio_ret - risk_free_rate -  2 * portfolio_vol
    return -value # Negate Sharpe ratio because we minimize the function


def min_func_one_sigma(weights, returns, covariance, risk_free_rate):
  portfolio_ret = portfolio_log_return(weights, returns)
  portfolio_vol = portfolio_volatility_log_return(weights, covariance)
  return -(portfolio_ret - risk_free_rate - portfolio_vol)



def optimize_portfolio(returns, covariance, risk_free_rate, bounds):
    num_assets = len(returns)
    args = (returns, covariance, risk_free_rate)

    # Define constraints
    def constraint_sum(weights):
        return np.sum(np.abs(weights)) - 1

    constraints = [{'type': 'eq', 'fun': constraint_sum}]


    # Perform optimization
    def objective(weights):
        return min_func_one_sigma(weights, returns, covariance, risk_free_rate)

    iteration = [0]  # mutable container to store iteration count
    def callback(weights):
        iteration[0] += 1

        print(f"Iteration: {iteration[0]}, value: {objective(weights)}")

    # Initial guess (equal weights)
    initial_guess = num_assets * [1. / num_assets]

    # Perform optimization
    result = minimize(objective, initial_guess,
                      method='SLSQP', bounds=bounds, constraints=constraints, callback=callback, options={'maxiter': 100})

    return result

def get_bounds(tickers, lower_bound, upper_bound):
  # for ETF, the allowed weight is between 0 and 20%
  # for stocks, the allowed weight is between 0 and 10%
  num_assets = len(tickers)
  if num_assets == 0:
    return None

  bounds = tuple((lower_bound, upper_bound) for ticker in tickers)
  return bounds


def do_optimization(mu, S, final_tickers, period, upper_bound):
  riskfree_log_return = np.log(1 + INTEREST_RATE) * period / ANNUAL_TRADING_DAYS
  bounds = get_bounds(final_tickers, 0, upper_bound)
  raw_weights = optimize_portfolio(mu, S, riskfree_log_return, bounds)
  weights = raw_weights.x

  tickers_to_buy = []
  for index, ticker_name in enumerate(final_tickers):
    weight = weights[index]
    if weight > 1e-3 or weight < -1e-3:
      logger.info(f'index: {index} {ticker_name}: weight {weight} exp profit: {mu[index]}, variance: {S[ticker_name][ticker_name]}')
      ticker_info = {'id': ticker_name, 'weight': weight}
      tickers_to_buy.append(ticker_info)

  logger.info(f'expected return in {period} trading days: {portfolio_return(weights, mu)}')
  logger.info(f'volatility of the return in {period} trading days: {portfolio_volatility(weights, S)}')
  # print tickers_to_buy in JSON format
  return tickers_to_buy




def get_shrinkage_covariance(df):
    lw = LedoitWolf(store_precision=False, assume_centered=True)
    lw.fit(df)
    # Convert the ndarray back to a DataFrame and use the column and index from the original DataFrame
    shrink_cov = pd.DataFrame(lw.covariance_, index=df.columns, columns=df.columns)
    return shrink_cov


def get_doubled_matrix(S):
    m = S.shape[0]  # Assuming S is a square matrix

    # Create the numpy array for S_prime
    S_prime_array = np.zeros((2*m, 2*m))
    S_prime_array[:m, :m] = S
    S_prime_array[m:, m:] = S
    S_prime_array[:m, m:] = -S
    S_prime_array[m:, :m] = -S

    # Get the original column and index names
    original_columns = S.columns
    original_index = S.index

    # Create new column and index names
    new_columns = list(original_columns) + [f'-{col}' for col in original_columns]
    new_index = list(original_index) + [f'-{idx}' for idx in original_index]

    # Create the new DataFrame with appropriate column and index names
    S_prime = pd.DataFrame(S_prime_array, columns=new_columns, index=new_index)

    return S_prime


def get_errors_mu_short(all_errors, mu):
  # for those stocks that have negative log return in mu, we need to reverse the sign of the error
  errors = all_errors.copy()
  # Iterate over mu and all_errors together using enumerate
  for i, log_return in enumerate(mu):
      # Check if the log return is negative
      if log_return < 0:
          stock_name = all_errors.columns[i]
          # Reverse the sign of the corresponding error
          errors[stock_name] = -errors[stock_name]

  return errors, np.abs(mu)


def save_json_to_dir(ticket_to_buy_json, out_dir):
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  cur_date = pd.Timestamp.now().strftime('%Y%m%d')
  with open(f'{out_dir}/ticket_to_buy_{cur_date}.json', 'w') as f:
    json.dump(ticket_to_buy_json, f)


def update_stock_operation_and_weight(stock, index, mu):
    """
    Updates the operation and weight of a stock based on the mu value.
    """
    if mu[index] < 0:
        stock['operation'] = 'short'
        stock['weight'] = -stock['weight']
    else:
        stock['operation'] = 'long'
    return stock['weight']