import numpy as np
import matplotlib.pyplot as plt

def compound_interest(principal, rate, years):
    # principal: initial amount of money
    # rate: annual interest rate
    # years: no. of years
    
    return principal * (1 + rate) ** years

def continuous_compound_interest(principal, rate, years):
    
    return principal * exp(rate * years)

def portfolio_stats(weights, mean_returns, cov_matrix):
    weights = np.array(weights)
    weights = weights / np.sum(weights)
    
    expected_return = weights @ mean_returns
    volatility = np.sqrt(weights @ cov_matrix @ weights)
    
    return expected_return, volatility

def portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate=0.0):
    """
    Compute expected return, volatility, and Sharpe ratio.

    Parameters:
    weights (np.ndarray): portfolio weights
    mean_returns (np.ndarray): expected returns
    cov_matrix (np.ndarray): covariance matrix
    risk_free_rate (float): risk-free rate

    Returns:
    dict: {
        "expected_return": ...,
        "volatility": ...,
        "sharpe_ratio": ...
    }
    """
    # 1. Normalize weights (optional)
    
    weights = np.array(weights)
    weights = weights / np.sum(weights)

    # 2. Compute expected portfolio return
    
    expected_return = weights @ mean_returns

    # 3. Compute portfolio volatility
    
    volatility = np.sqrt(weights @ cov_matrix @ weights)

    # 4. Compute Sharpe ratio
    
    sharpe_ratio = (expected_return - risk_free_rate) / volatility

    # 5. Return as dictionary
    
    return {
    "expected_return": expected_return,
    "volatility": volatility,
    "sharpe_ratio": sharpe_ratio
}

def simulate_portfolios(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0.0):
    """
    Simulate random portfolios and compute metrics.

    Parameters:
    mean_returns (np.ndarray): expected returns
    cov_matrix (np.ndarray): covariance matrix
    num_portfolios (int): number of random portfolios
    risk_free_rate (float): risk-free rate

    Returns:
    dict: {
        "returns": np.ndarray of expected returns,
        "volatilities": np.ndarray of volatilities,
        "sharpe_ratios": np.ndarray of Sharpe ratios,
        "max_sharpe": dict of portfolio metrics with max Sharpe ratio
    }
    """
    # 1. Initialize arrays to store returns, volatilities, and Sharpe ratios
    
    n_assets = len(mean_returns)
    
    returns = np.zeros(num_portfolios)
    volatilities = np.zeros(num_portfolios)
    sharpe_ratios = np.zeros(num_portfolios)
    

    # 2. Loop over num_portfolios:
    #    a. Generate random weights (normalize so sum=1)
    #    b. Compute portfolio metrics using portfolio_metrics()
    #    c. Store results in arrays
    
    for n in range(num_portfolios):
        
        weights = np.random.rand(n_assets)
        weights = weights / np.sum(weights)
        metrics = portfolio_metrics(weights, mean_returns, cov_matrix, risk_free_rate)
        
        returns[n] = metrics["expected_return"]
        volatilities[n] = metrics["volatility"]
        sharpe_ratios[n] = metrics["sharpe_ratio"]
        
    # 3. Identify portfolio with maximum Sharpe ratio
    
    max_index = np.argmax(sharpe_ratios)
    
    max_sharpe = {
        "expected_return": returns[max_index],
        "volatility": volatilities[max_index],
        "sharpe_ratio": sharpe_ratios[max_index]
    }

    # 4. Return arrays + max Sharpe portfolio metrics
    
    return {
        "returns": returns,
        "volatilities": volatilities,
        "sharpe_ratios": sharpe_ratios,
        "max_sharpe": max_sharpe
    }

def plot_efficient_frontier(simulation_results):
    """
    Plot the efficient frontier from simulated portfolios.

    Parameters:
    simulation_results (dict): output of simulate_portfolios()

    Returns:
    None
    """
    # 1. Extract returns, volatilities, and Sharpe ratios
    returns = simulation_results["returns"]
    volatilities = simulation_results["volatilities"]
    sharpe_ratios = simulation_results["sharpe_ratios"]

    # 2. Create a scatter plot of portfolios
    #    X-axis: volatilities
    #    Y-axis: returns
    #    Color: Sharpe ratios
    plt.scatter(volatilities, returns, c = sharpe_ratios, cmap = "viridis", alpha = 0.5)

    # 3. Highlight the portfolio with maximum Sharpe ratio
    max_sharpe = simulation_results["max_sharpe"]
    max_return = max_sharpe["expected_return"]
    max_volatility = max_sharpe["volatility"]
    
    plt.scatter(max_volatility, max_return, color = "red", marker = "*", s = 200)

    # 4. Add labels, title, and colorbar
    
    plt.xlabel("Volatility")
    plt.ylabel("Expected return")
    plt.title("Efficient Frontier")
    plt.colorbar(label = "Sharpe ratio")

    # 5. Display the plot
    
    plt.show()

def monte_carlo_portfolio(
    initial_value, expected_return, volatility, days=252, num_simulations=1000
):
    """
    Simulate random portfolio value paths over time.

    Parameters:
    initial_value (float): starting portfolio value
    expected_return (float): expected annual return
    volatility (float): annual volatility
    days (int): number of trading days to simulate
    num_simulations (int): number of Monte Carlo paths

    Returns:
    np.ndarray: simulated portfolio values of shape (days, num_simulations)
    """
    # 1. Convert annual return/vol to daily equivalents
    
    daily_return = (1 + expected_return) ** (1 / 252) - 1
    daily_volatility = (volatility) / np.sqrt(252)
    
    # 2. Generate random daily returns (normal noise)
    
    random_daily_returns = np.random.normal(daily_return, daily_volatility, (days, num_simulations))
        
    # 3. Compute cumulative product to get price paths
    
    portfolio_paths = initial_value * np.cumprod(1 + random_daily_returns, axis = 0)
    
    # 4. Return simulated paths
    
    return portfolio_paths

import matplotlib.pyplot as plt

def plot_monte_carlo_paths(portfolio_paths, num_paths_to_plot=10):
    """
    Plot sample Monte Carlo portfolio paths.

    Parameters:
    portfolio_paths (np.ndarray): simulated portfolio values (days x simulations)
    num_paths_to_plot (int): how many random paths to plot

    Returns:
    None
    """
    # 1. Select random subset of paths to plot
    #    (Hint: np.random.choice on the columns)
    
    random_paths = np.random.choice(portfolio_paths.shape[1], num_paths_to_plot, replace = False)

    # 2. Plot each selected path
    
    for n in random_paths:
        plt.plot(portfolio_paths[:, n], alpha = 0.7)

    # 3. Add labels and title
    
    plt.xlabel("Time (days)")
    plt.ylabel("Portfolio Value")
    plt.title("Monte Carlo paths of portfolio")

    # 4. Show plot
    
    plt.show()

def analyse_monte_carlo_results(portfolio_paths, confidence_level=0.05):
    """
    Analyze Monte Carlo portfolio simulations.

    Parameters:
    portfolio_paths (np.ndarray): simulated portfolio values (days x simulations)
    confidence_level (float): e.g. 0.05 for 5% Value at Risk

    Returns:
    dict: {
        "mean_final_value": ...,
        "std_final_value": ...,
        "VaR": ...,
        "CVaR": ...
    }
    """
    # 1. Extract final portfolio values (last row)
    
    final_values = portfolio_paths[-1, :]

    # 2. Compute mean and std of final values
    
    mean_fv = np.mean(final_values)
    std_fv = np.std(final_values)

    # 3. Compute Value at Risk (VaR)
    #    Hint: np.percentile(final_values, confidence_level * 100)
    
    v0 = portfolio_paths[0, 0]
    
    var = v0 - np.percentile(final_values, confidence_level * 100)
    
    # 4. Compute Conditional VaR (CVaR, aka Expected Shortfall)
    #    Hint: mean of all values below the VaR threshold
    
    worst_outcomes = final_values[final_values <= v0 - var]
    
    cvar = v0 - np.mean(worst_outcomes)

    # 5. Return dictionary of results
    
    return {
        "mean": mean_fv,
        "standard_deviation": std_fv,
        "value_at_risk": var,
        "expected_shortfall": cvar
    }
