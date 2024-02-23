import numpy as np
import pandas as pd

class backTester:
    def __init__(self):
        pass
    def roll(self,data,model,train_size = pd.DateOffset(months=9),test_size = pd.DateOffset(months=3),logger=False,**model_params):
        """
        backtest the strategy on time series model
        :param data: the dataset, should be a time series with pd.datetime as index
        :param model: the model we want to use. Must have a member function model.evaluate_strategy()
        :param train_size: size of training set
        :param test_size: size of test set
        :param logger: if log the useful information
        :param model_params: a dictionary that contains necessary model parameters
        :return:
        """
        start_time = data.index[0]
        end_time = data.index[-1]
        # the start time of the training set
        curr_time = start_time
        pnl = pd.DataFrame({})
        while curr_time + train_size < end_time:
            if logger:
                print(f'current time: {curr_time}')
            # split train data and test data
            train_data = data.loc[(data.index >= curr_time) & (data.index < curr_time + train_size)].copy()

            if train_data.empty:
                # if the training set is empty (for instance, get a trading time at night), then proceed to the next window
                curr_time = curr_time + test_size
                continue

            test_data = data.loc[(data.index >= curr_time + train_size) & (data.index < curr_time + train_size + test_size)].copy()

            if test_data.empty:
                # if the test set is empty (for instance, get some time at night), proceed to the next window
                curr_time = curr_time + test_size
                continue

            # calculate total return by model.evaluate_strategy()
            # notice that model.evaluate_strategy() contains both training and testing, should be defined by user
            total_returns = model.evaluate_strategy(train_data, test_data)
            # due to some reasons, the model is not executed (for instance, no correlated pairs in pair trading)
            if len(total_returns) == 0 or (total_returns is None):
                # if no pnl curves (no pairs found), fill them by 1
                curr_pnl = pd.DataFrame(1, index=test_data.index, columns=pnl.columns)
            else:
                # otherwise, get returns first and summarize them (assume equal position)
                curr_pnl = (1 + total_returns)
                #curr_pnl = np.mean(curve, axis=0)

            # add new returns to the current returns
            #pnl = np.concatenate((pnl, curr_pnl))
            pnl = pd.concat([pnl, curr_pnl], axis=0, ignore_index=True, sort=False)
            # update the current date
            curr_time = curr_time + test_size
        # get pnl, fill na with 1 (not earning or losing)
        pnl = pnl.fillna(1)
        pnl = np.cumprod(pnl)
        return pnl

    def metrics(self,pnl):
        """
        evaluate the metrics of a pnl curve.
        :param pnl: a pnl curve
        :return: return a dictionary that contains the metrics
        """
        index = {}
        risk_free_rate = 0

        # Convert the input to a numpy array if it's not already one
        pnl_curve = np.array(pnl)

        # Calculate daily returns
        daily_returns = np.diff(pnl_curve) / pnl_curve[:-1]

        # Sharpe Ratio calculation
        mean_daily_return = np.mean(daily_returns)
        std_daily_return = np.std(daily_returns)

        # Annualizing the Sharpe Ratio assuming the number of data points corresponds to the number of trading days
        sharpe_ratio = (mean_daily_return - risk_free_rate / 252) / std_daily_return * np.sqrt(252)

        # Max Drawdown calculation
        cumulative_returns = np.cumprod(1 + daily_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)

        index['Sharpe Ratio'] = sharpe_ratio
        index['Max Drawdown'] = max_drawdown

        return index
