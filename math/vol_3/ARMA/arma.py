# Name
# Date
# Class

from concurrent.futures.process import _threads_wakeups
from scipy.stats.distributions import norm
from scipy.optimize import fmin
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pydataset import data as pydata
from statsmodels.tsa.stattools import arma_order_select_ic as order_select
import pandas as pd
from scipy.optimize import minimize


def kalman(F, Q, H, time_series):
    # Get dimensions
    dim_states = F.shape[0]
    # Initialize variables
    # covs[i] = P_{i | i-1}
    covs = np.zeros((len(time_series), dim_states, dim_states))
    mus = np.zeros((len(time_series), dim_states))
    # Solve of for first mu and cov
    covs[0] = np.linalg.solve(np.eye(dim_states**2) - np.kron(F,F),np.eye(dim_states**2)).dot(Q.flatten()).reshape(
            (dim_states,dim_states))
    mus[0] = np.zeros((dim_states,))
    # Update Kalman Filter
    for i in range(1, len(time_series)):
        t1 = np.linalg.solve(H.dot(covs[i-1]).dot(H.T),np.eye(H.shape[0]))
        t2 = covs[i-1].dot(H.T.dot(t1.dot(H.dot(covs[i-1]))))
        covs[i] = F.dot((covs[i-1] - t2).dot(F.T)) + Q
        mus[i] = F.dot(mus[i-1]) + F.dot(covs[i-1].dot(H.T.dot(t1))).dot(
                time_series[i-1] - H.dot(mus[i-1]))
    return mus, covs
def state_space_rep(phis, thetas, mu, sigma):
    # Initialize variables
    dim_states = max(len(phis), len(thetas)+1)
    dim_time_series = 1 #hardcoded for 1d time_series
    F = np.zeros((dim_states,dim_states))
    Q = np.zeros((dim_states, dim_states))
    H = np.zeros((dim_time_series, dim_states))
    # Create F
    F[0][:len(phis)] = phis
    F[1:,:-1] = np.eye(dim_states - 1)
    # Create Q
    Q[0][0] = sigma**2
    # Create H
    H[0][0] = 1.
    H[0][1:len(thetas)+1] = thetas
    return F, Q, H, dim_states, dim_time_series

def arma_forecast_naive(file='weather.npy',p=2,q=1,n=20):
    """
    Perform ARMA(1,1) on data. Let error terms be drawn from
    a standard normal and let all constants be 1.
    Predict n values and plot original data with predictions.

    Parameters:
        file (str): data file
        p (int): order of autoregressive model
        q (int): order of moving average model
        n (int): number of future predictions
    """
    data = np.load(file)
    y = np.diff(data, 1)
    phi = .5
    theta = .1
    c = 0
    eps = list(np.random.normal(0, 1, q+1))
    preds = list(y[-p:])
    for t in range(n):
        AR = sum([phi * preds[t+i] for i in range(p)])
        MA = sum([theta * eps[t - j] for j in range(1, q+1)]) + eps[-1]
        preds.append(c + AR + MA)
        eps.append(np.random.normal(0, 1))
    return y, np.array(preds[p:])

# y, preds = arma_forecast_naive()
# plt.plot(y)
# plt.plot(np.arange(len(y), len(y) + len(preds)), preds)
# plt.show()

def arma_likelihood(file='weather.npy', phis=np.array([0.9]), thetas=np.array([0]), mu=17., std=0.4):
    """
    Transfer the ARMA model into state space. 
    Return the log-likelihood of the ARMA model.

    Parameters:
        file (str): data file
        phis (ndarray): coefficients of autoregressive model
        thetas (ndarray): coefficients of moving average model
        mu (float): mean of errorm
        std (float): standard deviation of error

    Return:
        log_likelihood (float)
    """
    data = np.load(file)
    data = np.diff(data)
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)
    mus, covs = kalman(F, Q, H, data)
    # print(covs[0])
    k1 = np.array([H@mus[i] + mu for i in range(len(mus))])
    k2 = np.array([H @ covs[i] @ H.T for i in range(len(covs))])

    # print(k1, k2)
    data = np.array([np.log(norm.pdf(data[i],k1[i],k2[i]) + 3.85e-9) for i in range(len(mus))])
    # data = [i for i in data.flatten() if str(i) != "nan"]
    return np.sum(data)

def model_identification(file='weather.npy',p=4,q=4):
    """
    Identify parameters to minimize AIC of ARMA(p,q) model

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model

    Returns:
        phis (ndarray (p,)): coefficients for AR(p)
        thetas (ndarray (q,)): coefficients for MA(q)
        mu (float): mean of error
        std (float): std of error
    """
    best_vals = [0, 0, 0, 0]
    best_aic = 1e99
    time_series = np.load(file)
    def f(x): # x contains the phis, thetas, mu, and std
        return -1*arma_likelihood(file, phis=x[:p], thetas=x[p:p+q],mu=x[-2],std=x[-1])
    # create initial point
    for i in range(1, p+1):
        for j in range(1, q+1):
            x0 = np.zeros(i+j+2)
            x0[-2] = time_series.mean()
            x0[-1] = time_series.std()
            sol = minimize(f,x0, method = "SLSQP")
            sol = sol['x']
            phis = sol[:i]
            thetas = sol[i:i+j]
            mu = sol[-2]
            std = sol[-1]
            l = arma_likelihood(file, phis, thetas, mu, std)
            n = len(time_series)
            k = i + j + 2
            aic = 2*k * (1 + (k+1) / (n-k)) - 2*l
            if aic < best_aic:
                best_vals = [phis, thetas, mu, std]
    return best_vals
# print(model_identification())

def arma_forecast(file='weather.npy', phis=np.array([0]), thetas=np.array([0]), mu=0., std=0., n=30):
    """
    Forecast future observations of data.
    
    Parameters:
        file (str): data file
        phis (ndarray (p,)): coefficients of AR(p)
        thetas (ndarray (q,)): coefficients of MA(q)
        mu (float): mean of ARMA model
        std (float): standard deviation of ARMA model
        n (int): number of forecast observations

    Returns:
        new_mus (ndarray (n,)): future means
        new_covs (ndarray (n,)): future standard deviations
    """
    data = np.diff(np.load(file))
    F, Q, H, dim_states, dim_time_series = state_space_rep(phis, thetas, mu, std)
    mus, covs = kalman(F, Q, H, data)
    new_mus = [mus[[-1]].flatten()]
    new_covs = [covs[[-1]].flatten().reshape(2, 2)]

    for i in range(n):
        
        new_mus.append(F @ new_mus[-1].T + mu)
        new_covs.append(F @ new_covs[-1] @ F.T + Q)

    new_mus = np.array(new_mus)
    new_covs = np.array(new_covs)
    plt.plot(data)
    domain = np.arange(len(data), len(data) + 30)
    plt.plot(domain, np.mean(new_mus[1:], axis=1))
    plt.plot(domain, np.mean(new_mus[1:], axis=1) + 2*np.sqrt(std), c='tab:green')
    plt.plot(domain, np.mean(new_mus[1:], axis=1) - 2*np.sqrt(std), c='tab:green')
    plt.show()

    return new_mus, new_covs
        


# # Get optimal model as found in the previous problem
# phis, thetas, mu, std = np.array([ 0.72135856]), np.array([-0.26246788]), 0.35980339870105321, 1.5568331253098422
# # Forecast optimal mode
# new_mus, new_covs = arma_forecast(file='weather.npy', phis=phis, thetas=thetas, mu=mu, std=std)


def sm_arma(file = 'weather.npy', p=4, q=4, n=30):
    """
    Build an ARMA model with statsmodel and 
    predict future n values.

    Parameters:
        file (str): data file
        p (int): maximum order of autoregressive model
        q (int): maximum order of moving average model
        n (int): number of values to predict

    Return:
        aic (float): aic of optimal model
    """
    best_model = 0
    best_aic = 1e99
    data = np.diff(np.load(file))
    for pp in range(p):
        for qq in range(q):
            model = ARIMA(data,order=(p,0,q),trend='c').fit(method='innovations_mle')
            if model.aic < best_aic:
                best_model = model
        
    preds = best_model.predict(start=0,end=len(data)+30)
    plt.plot(data)
    plt.plot(preds)
    plt.show()
sm_arma()

def manaus(start='1983-01-31',end='1995-01-31',p=4,q=4):
    """
    Plot the ARMA(p,q) model of the River Negro height
    data using statsmodels built-in ARMA class.

    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting
        p (int): max_ar parameter
        q (int): max_ma parameter
    Return:
        aic_min_order (tuple): optimal order based on AIC
        bic_min_order (tuple): optimal order based on BIC
    """
    # Get dataset
    raw = pydata('manaus')
    # Make DateTimeIndex
    manaus = pd.DataFrame(raw.values,index=pd.date_range('1903-01','1993-01',freq='M'))
    manaus = manaus.drop(0,axis=1)
    # Reset column names
    manaus.columns = ['Water Level']
