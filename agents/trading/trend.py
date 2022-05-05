import numpy as np


def centered_moving_average(timeseries_df, window_size):
    """
    Moving average over a window centered on the current data point
    """
    if window_size % 2 == 0:
        shift = int(window_size/2)
    else:
        shift = int((window_size - 1)/2)
    timeseries_df.loc[:, f'cm_average_{window_size}'] = timeseries_df.rolling(window_size).mean().shift(-shift)
    return timeseries_df


def exponential_moving_average(timeseries_df, window_size):
    timeseries_df.loc[:, f'ewm_{window_size}'] = timeseries_df.ewm(span=window_size).mean()
    return timeseries_df


def daily_volatility(full_history_high, full_history_low):
    # (Hi - Low)/2 / (Hi + Low)/2
    return ((full_history_high - full_history_low) /
            (full_history_high + full_history_low)).mean()


def volatility(timeseries_df, window_size, price_col):
    df = centered_moving_average(timeseries_df, window_size)
    return np.sqrt(((df[f'cm_average_{window_size}']/df[price_col] - 1)**2).mean())


def trend_margins(full_history, window_size):
    """
    Upper and lower conservative margins over the ewa,
    calculated as the quadrature sum of the daily_volatility and the offset of ewa from the centered moving average
    """
    daily_var = daily_volatility(full_history['High'], full_history['Low'])**2

    timeseries_df = exponential_moving_average(full_history[['Open']], window_size).join(
        centered_moving_average(full_history[['Open']], window_size).drop('Open', axis=1)
    )
    ews_var = ((timeseries_df[f'ewm_{window_size}'] / timeseries_df[f'cm_average_{window_size}'] - 1) ** 2).mean()
    return np.sqrt(ews_var + daily_var)
