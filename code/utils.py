import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats import t
import arch
import matplotlib.pyplot as plt

from pypfopt import risk_models, black_litterman
from pypfopt.black_litterman import BlackLittermanModel
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import objective_functions
import cvxpy as cp

#--------------------------------------------------------------------------------- MATRIX-GENERATING FUNCTIONS --------------------------------------------------------------------

def Q_average(tr):
    # return average of outer product of [eT-1,...e0]
    # et = [r(1t)/s(1t),...r(nt)/s(nt)]
    T = tr.shape[1]
    n = tr.shape[0]
    sum = np.zeros([n,n])
    for i in range(T):
        sum += np.outer(tr[:,i],tr[:,i])
    return sum/T

def Q_gen(tr,ab):
    # generate [Q0,...QT-1] -- assume Q0 = Q_int
    Q_int = Q_average(tr)
    Q_list = [Q_int]
    T = tr.shape[1]
    a = ab[0]
    b = ab[1]
    for i in range(1,T):
        et_1 = tr[:,i-1]
        Qt_1 = Q_list[i-1]
        Qt = (1.0-a-b)*Q_int + a*np.outer(et_1,et_1) + b*Qt_1
        Q_list = Q_list + [Qt]
    return Q_list

def R_gen(tr,ab): #Qt --> Rt
    # output [R0,...RT-1]
    Q_list = Q_gen(tr,ab)
    R_list = []
    n = Q_list[0].shape[0]
    for Qt in Q_list:
        Q_star = np.sqrt(np.diag(1/np.diag(Qt + 1e-10)))
        Rt = Q_star @ Qt @ Q_star
        R_list = R_list + [Rt]
    return R_list

def D_gen(sigma):
    D_list = np.array([np.diag(sigma[:, i]) for i in range(sigma.shape[1])])
    return D_list

def H_gen(D, R):
    H_list = np.array([np.dot(np.dot(D[i], R[i]), D[i]) for i in range(len(R))])
    return H_list

def H_sqrt_gen(H):
    H_sqrt_list = np.array([np.linalg.cholesky(H[i]) for i in range(len(H))])
    return H_sqrt_list


#----------------------------------------------------------------------------- CORRELATION-MATRIX FUNCTIONS -----------------------------------------------------------------------------------

def vecl(matrix):
    lower_matrix = np.tril(matrix,k=-1)
    array_with_zero = np.matrix(lower_matrix).A1

    array_without_zero = array_with_zero[array_with_zero!=0]

    return array_without_zero

def Corr_data(tr, ab):
    N, T = int(tr.shape[0]), int(tr.shape[1])
    veclRt =  np.zeros((T, int(N*(N-1)/2)))
    Rt= R_gen(tr,ab)
    for j in range(0,T):
      veclRt[j, :] = vecl(Rt[j])
    return veclRt

#------------------------------------------------------------------------------ LOSS-FUNCTIONS -------------------------------------------------------------------------------------------------

# This is loss function
def loglike_norm_dcc_copula(ab, udata):
    N, T = int(udata.shape[0]), int(udata.shape[1])
    llf = np.zeros((T,1))
    trdata = np.array(norm.ppf(udata), ndmin=2)

    Rt =  R_gen(trdata, ab)

    for i in range(0,T):
        llf[i] = -0.5* np.log(np.linalg.det(Rt[i]))
        llf[i] = llf[i] - 0.5 *  np.matmul(np.matmul(trdata[:,i] , (np.linalg.inv(Rt[i]) - np.eye(N))) ,trdata[:,i].T)
    llf = np.sum(llf)

    return -llf

#------------------------------------------------------------------------------- COPULA-FUNCTIONS ----------------------------------------------------------------------------------------------

# Transform t-distribution to a uniform distribution
def garch_t_to_u(rets, garch_results):
    mu = garch_results.params['mu']
    nu = garch_results.params['nu']
    est_r = rets - mu
    h = garch_results.conditional_volatility
    std_res = est_r / h
    std_res = garch_results.std_resid
    udata = t.cdf(std_res, nu)
    return udata

class DCC():

    def __init__(self, max_itr=2):
        self.max_itr = max_itr
        self.ab = np.array([0.05, 0.95])
        self.method =  'SLSQP'
        self.Qt = None
        self.Rt = None
        self.Dt = None
        self.Ht = None
        self.H_sqrt_t = None
        self.loss_func = None
        def ub(x):
            return 1. - x[0] - x[1]
        def lb1(x):
            return x[0]
        def lb2(x):
            return x[1]
        self.cons = [{'type':'ineq', 'fun':ub},{'type':'ineq', 'fun':lb1},{'type':'ineq', 'fun':lb2}]
        self.bnds = ((0, 0.5), (0, 0.9997))
        self.epsilon = None

    def set_ab(self,ab): # ndarray
        self.ab = ab

    def set_method(self,method):
        self.method = method

    def set_loss(self, loss_func):
        self.loss_func = loss_func

    def get_loss_func(self):
        if self.loss_func is None:
            raise Exception("No Loss Function Found!")
        else:
            return self.loss_func

    def run_garch_on_return(self, rets):
        udata_list = []
        model_parameters = {}
        for x in rets:
            am = arch.arch_model(rets[x], vol='Garch', p=1, q=1, o=1, dist = 't')
            stock_code = x.split('_')[0]
            model_parameters[stock_code] = am.fit(disp='off')
            udata = garch_t_to_u(rets[x], model_parameters[stock_code])
            udata_list.append(udata)
        udata_list = np.array(udata_list)
        return udata_list, model_parameters

    def fit(self, train_data):

        udata_list, model_parameters = self.run_garch_on_return(train_data)

        tr = udata_list

        # Optimize using scipy and save theta
        tr_losses = []
        j = 0
        count = 0
        while j < self.max_itr:
            j += 1
            ab0 = np.array(self.ab)
            res = minimize(self.get_loss_func(), ab0, args = (tr,), bounds=self.bnds, constraints=self.cons)
            ab = res.x
            self.set_ab(ab)

            tr_loss = self.get_loss_func()(ab,tr)
            tr_losses.append(tr_loss)

        print("Successfully Trained!")

        epsilon = np.array(norm.ppf(udata_list), ndmin=2)
        sigma = np.array([np.array(model_parameters[i.split('_')[0]].conditional_volatility) for i in train_data])
        self.epsilon = epsilon
        self.Ht = self.H_t(epsilon, sigma)

        return tr_losses

    def Ht_forecast(self, test_data):
        udata_list, model_parameters = self.run_garch_on_return(test_data)
        epsilon = np.array(norm.ppf(udata_list), ndmin=2)
        sigma = np.array([np.array(model_parameters[i.split('_')[0]].conditional_volatility) for i in test_data])
        H_t_forecasted = self.H_t(epsilon, sigma)
        return H_t_forecasted

    def Q(self,y):
        self.Qt = np.array(Q_gen(y,self.ab))
        return self.Qt

    def Q_bar(self,y):
        return Q_average(y)

    def R_t(self, y):
        self.Rt = np.array(R_gen(y,self.ab))
        return self.Rt

    def D_t(self, sigma):
        self.Dt = D_gen(sigma)
        return self.Dt

    def H_t(self, y, sigma):
        Dt = np.array(D_gen(sigma))
        Rt = np.array(R_gen(y,self.ab))
        Ht = H_gen(Dt, Rt)
        return Ht

    def H_sqrt_t(self, y, sigma):
        self.Dt = D_gen(sigma)
        self.Rt = np.array(R_gen(y,self.ab))
        self.Ht = H_gen(self.Dt, self.Rt)
        self.H_sqrt_t = H_sqrt_gen(self.Ht)
        return self.H_sqrt_t

def portfolio_liquidity(weights, liquidity_scores):
    return sum(weights[i] * score for i, score in enumerate(liquidity_scores.values()))

def Port_period_perfomance(port_perfomance):
    expected_return = port_perfomance[0]
    expected_vol = port_perfomance[1]
    periods = [6, 12, 20, 30]
    min_prob = [0.9, 0.8, 0.7, 0.6]
    t_stat = [1.28, 0.84, 0.52, 0.25 ]
    _dict = {'6': None, '12': None, '20': None, '30': None}
    for period in periods:
      for i in range(len(min_prob)):
          expected_return_dis = expected_return*period
          vol_dis = expected_vol*np.sqrt(period)
          required_return = (expected_return_dis-vol_dis*t_stat[i])/period
          if _dict[str(period)] == None:
            _dict[str(period)] = [round(required_return*100,2)]
          else:
            _dict[str(period)].append(round(required_return*100,2))
    return _dict

def portfolio_os_testing(weights, returns):
    weight_array = np.array(list(weights.values()))
    portfolio_returns = returns.dot(weight_array)
    portfolio_volatility = portfolio_returns.std()*np.sqrt(252)
    portfolio_sharpe = portfolio_returns.mean()*252/portfolio_volatility
    return portfolio_sharpe, portfolio_returns, portfolio_volatility


def monte_carlo_simulation(e_returns, cov_matrix, weight, num_simulations=10000):

    # 1. Define the Inputs

    mu = e_returns # Expected returns for each asset (example)
    Sigma = cov_matrix  # Covariance matrix
    w = np.array(list(weight.values())) # Portfolio weights
    N = num_simulations # Number of simulations


    # Constraint: Asset [-1] (i.e., the last asset) return should not fall below 0
    threshold = 0

    # 2. Generate Random Portfolio Returns with Constraint
    simulated_returns = []

    for _ in range(N):
        # Generate a random vector of returns for the assets
        r = np.random.multivariate_normal(mu, Sigma)

        # Check if the return of the last asset falls below the threshold
        while r[-1] < threshold:
            r = np.random.multivariate_normal(mu, Sigma)

        # Append valid returns to the list
        simulated_returns.append(r)

    # Convert to NumPy array for convenience
    simulated_returns = np.array(simulated_returns)

    # 3. Calculate Portfolio Returns for each simulation
    portfolio_returns = np.dot(simulated_returns, w)

    # 4. Analyze Results
    mean_portfolio_return = np.mean(portfolio_returns)
    std_portfolio_return = np.std(portfolio_returns)

    # Print summary statistics
    print(f"Mean Portfolio Return: {mean_portfolio_return:.4f}")
    print(f"Portfolio Return Standard Deviation: {std_portfolio_return:.4f}")

    # 5. Plot the Distribution of Portfolio Returns
    plt.hist(portfolio_returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.title('Monte Carlo Simulation of Portfolio Returns')
    plt.xlabel('Portfolio Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# @title
def aggregate_portfolios(proportions, *weights):

    # Check that proportions sum to 1 (100%)
    if sum(proportions) != 1:
        raise ValueError("Proportions must sum to 1.")

    # Initialize the aggregated portfolio with zero values
    aggregated_weights = OrderedDict()

    # Iterate over each portfolio and its corresponding proportion
    for i, weight_dict in enumerate(weights):
        if i == 0:
            # Initialize aggregated_weights with the first portfolio
            aggregated_weights = OrderedDict((key, value * proportions[i])
                                             for key, value in weight_dict.items())
        else:
            # Update the aggregated_weights by adding the weighted portfolio values
            for key, value in weight_dict.items():
                aggregated_weights[key] += value * proportions[i]
    aggregated_weights = OrderedDict((key, round(value,4)) for key, value in aggregated_weights.items())

    return aggregated_weights


def portfolio_annualized_metrics(weights, expected_returns, cov_matrix):

    # Extract the values from the OrderedDict (weights)
    weight_array = np.array(list(weights.values()))
    asset_names = list(weights.keys())

    # Ensure expected_returns is in the correct format (numpy array)
    expected_returns = np.array(expected_returns)

    # Calculate the annualized expected return (weighted sum of individual asset returns)
    annualized_return = np.dot(weight_array, expected_returns)

    # Calculate the portfolio variance (w^T * Covariance Matrix * w)
    portfolio_variance = np.dot(weight_array.T, np.dot(cov_matrix, weight_array))

    # Calculate annualized volatility (sqrt of portfolio variance)
    annualized_volatility = np.sqrt(portfolio_variance)

    return annualized_return, annualized_volatility


def optimal_asset_weights(price_data = None, views = None, market_cap= None):

    covar_matrix = risk_models.sample_cov(price_data)

    # Initialize the Black-Litterman model
    bl = BlackLittermanModel(
        covar_matrix,
        pi="market",
        absolute_views=views,
        market_caps=market_cap,
    )
    expected_returns = bl.bl_returns()

    # Initialize the Efficient Frontier
    ef = EfficientFrontier(expected_returns, covar_matrix)

    ef.add_constraint(lambda w: w >= 0.03)  # This ensures no asset has less than 2% weight

    # Optimize the portfolio by conditional drawdown
    weights = ef.max_sharpe()

    # Display results
    cleaned_weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=True)

    return cleaned_weights, performance, expected_returns

def portfolio_os_testing(weights, returns_os):
    weight_array = np.array(list(weights.values()))
    portfolio_returns = returns_os.dot(weight_array)
    portfolio_volatility = portfolio_returns.std()*np.sqrt(252)
    portfolio_sharpe = portfolio_returns.mean()*252/portfolio_volatility
    return portfolio_sharpe, portfolio_returns, portfolio_volatility, portfolio_returns.mean()*252


def value_at_risk(e_returns, cov_matrix, weight, num_simulations=10000):
    # 1. Define the Inputs
    mu = e_returns  # Expected returns for each asset
    Sigma = cov_matrix  # Covariance matrix
    w = np.array(list(weight.values()))  # Portfolio weights
    N = num_simulations  # Number of simulations

    # Constraint: Asset [-1] (i.e., the last asset) return should not fall below 0
    threshold = 0

    # 2. Generate Random Portfolio Returns with Constraint
    simulated_returns = []

    for _ in range(N):
        # Generate a random vector of returns for the assets
        r = np.random.multivariate_normal(mu, Sigma)

        # Check if the return of the last asset falls below the threshold
        while r[-1] < threshold:
            r = np.random.multivariate_normal(mu, Sigma)

        # Append valid returns to the list
        simulated_returns.append(r)

    # Convert to NumPy array for convenience
    simulated_returns = np.array(simulated_returns)

    # 3. Calculate Portfolio Returns for each simulation
    portfolio_returns = np.dot(simulated_returns, w)

    # 4. Analyze Results
    mean_portfolio_return = np.mean(portfolio_returns)
    std_portfolio_return = np.std(portfolio_returns)
    var_95 = np.percentile(portfolio_returns, 5)  # Calculate the 5th percentile

    # Print summary statistics
    print(f"Mean Portfolio Return: {mean_portfolio_return:.4f}")
    print(f"Portfolio Return Standard Deviation: {std_portfolio_return:.4f}")
    print(f"95% Value at Risk (VaR): {var_95:.4f}")

    # 5. Plot the Distribution of Portfolio Returns
    plt.hist(portfolio_returns, bins=50, alpha=0.75, color='blue', edgecolor='black')
    plt.axvline(var_95, color='red', linestyle='dashed', linewidth=1, label='95% VaR')
    plt.title('Monte Carlo Simulation of Portfolio Returns')
    plt.xlabel('Portfolio Return')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

def final_portfolios_os(aggregated_portfolio, return_os):
    weight_array = np.array(list(aggregated_portfolio.values()))
    total_return = return_os.dot(weight_array)
    return total_return

def calculate_security_weights(asset_allocation, security_selection):
    security_weights = {}

    # Iterate through each asset class
    for asset_class, asset_weight in asset_allocation.items():
        if asset_class in security_selection:
            # Get the security weights for this asset class
            security_class_weights = security_selection[asset_class]

            # Calculate the weight for each security
            for security, weight in security_class_weights.items():
                security_weights[security] = asset_weight * weight

    return security_weights

def plot_pie_chart(security_weights, threshold=1.0):
    # Prepare data for the pie chart
    securities = list(security_weights.keys())
    weights = list(security_weights.values())

    # Set a color palette using matplotlib (without seaborn)
    colors = plt.cm.Paired.colors  # Predefined color map "Paired"
    num_colors = len(securities)
    if num_colors > len(colors):
        colors = plt.cm.tab20c.colors  # Use tab20c if more than 12 categories

    # Create a new list for the labels that will include only significant ones
    labels = []
    percentages = []
    for sec, weight in zip(securities, weights):
        percentage = weight * 100
        if percentage > threshold:
            labels.append(f"{sec} ({percentage:.1f}%)")  # Ticker + Percentage
            percentages.append(percentage)
        else:
            labels.append('')  # Empty label for insignificant ones
            percentages.append(0)  # Empty percentage for insignificant ones

    # Plot the pie chart
    plt.figure(figsize=(10, 7))
    wedges, texts = plt.pie(
        percentages, labels=labels, startangle=140, colors=colors[:num_colors],
        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5}, pctdistance=0.85
    )

    # Title for the pie chart
    plt.title("Portfolio Security Weights", fontsize=16, fontweight='bold')

    # Customize the text style for labels (tickers + percentages)
    for text in texts:
        text.set_fontsize(10)  # Smaller tickers
        text.set_fontweight('bold')
        text.set_color('black')

    # Equal aspect ratio ensures that pie chart is circular
    plt.axis('equal')

    # Show the plot
    plt.show()
