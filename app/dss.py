# Tool 1
def tool_1():
  import pandas as pd
  # Import the CSV Data from AlphaVantage
  symbol = input("Please input the ticker symbol for your stock: ") # Request input values for stock symbol
  symbol = symbol.upper()
  stock_data = pd.read_csv(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={API_KEY}&datatype=csv')
  df = pd.DataFrame(stock_data)
  print(df) # Output the values for the stock

  import plotly.express as px # Visualize stock price over time
  stock_chart = px.line(df,x="timestamp",y="adjusted_close",title=f"{symbol} Stock Price Over Time",labels={"timestamp":"Date","adjusted_close":"Price"})
  stock_chart.show()

# Tool 2
def tool_2():
  import numpy as np # Import packages for math/statistical calculations & visualization
  from scipy.stats import norm
  import matplotlib.pyplot as plt
  import pandas as pd

  option = input("Please enter the ticker symbol for the option: ") # Obtain input values of stock name and strike price
  K = input("Please enter the strike price of your option: ")
  K = int(K)

  option_data = pd.read_csv(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={option}&apikey={API_KEY}&datatype=csv')
  df = pd.DataFrame(option_data) # Read stock values from API
  S = df["adjusted_close"][0] # Obtain current asset price
  df['returns'] = df["adjusted_close"].pct_change()
  sigma = df['returns'].std() * np.sqrt(252) # Obtain volatility of returns
  print(df)
  r_data = pd.read_csv(f'https://www.alphavantage.co/query?function=TREASURY_YIELD&interval=monthly&maturity=10year&apikey={API_KEY}&datatype=csv')
  df_r = pd.DataFrame(r_data)
  r = df_r["value"][0] / 100 # Obtain curent risk-free interest rate
  T = 1 # Set time to 1


  N = norm.cdf
  print(S,K,T,r,sigma) # List current asset price, strike price, time, risk-free interest rate, and volatility of returns
  def CallPrice(S, K, T, r, sigma): # Black-Scholes
    d_1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return S * N(d_1) - K * np.exp(-r*T)*N(d_2)
  def PutPrice(S, K, T, r, sigma):
    d_1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * N(-d_2) - S * N(-d_1)
  print("The value of the call is", "{:.2f}".format(CallPrice(S,K,T,r,sigma))) # Output values of call/put options at strike price
  print("The value of the put is","{:.2f}".format(PutPrice(S,K,T,r,sigma)))

  greeks_choice = input("View greeks?") # Ask user to display Greek option parameters
  if greeks_choice == "Y":
    print("GREEKS:")
    d_1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    OptionDelta = norm.cdf(d_1) # Calculate Greeks
    OptionGamma = norm.pdf(d_1)/(S*sigma*np.sqrt(T))
    OptionVega = S*np.sqrt(T)*norm.pdf(d_1)
    OptionTheta = -S*norm.pdf(d_1)*sigma/(2*np.sqrt(T)) - (r*K*np.exp(-r*T)*norm.cdf(d_2))
    OptionRho = K*T*np.exp(-r*T)*norm.cdf(d_2)
    print("Delta = ","{:.3f}".format(OptionDelta)) # Format output values
    print("Gamma = ","{:.3f}".format(OptionGamma))
    print("Vega = ","{:.3f}".format(OptionVega))
    print("Theta = ","{:.3f}".format(OptionTheta))
    print("Rho = ","{:.3f}".format(OptionRho))
    if OptionDelta >= 0:
      print("To hedge against this option's risk, please take a short position on","{:.0%}".format(OptionDelta),"of your total shares")
    else:
      print(print("To hedge against this option's risk, please take a long position on","{:.0%}".format(OptionDelta),"of your total shares"))
  elif greeks_choice == "N": # Data validity check
    print("Greeks not selected.")
  else:
    print("Invalid data input - please restart code.")

  sensitivity_choice = input("View call-put parity graph?") # Ask user to display visualization of option pricing
  if sensitivity_choice == "Y":
    S_min = input("Please input minimum asset price value: ") # Set minimum and maxmimum values for the x-axis
    S_min = int(S_min)
    S_max = input("Please input maximum asset price value: ")
    S_max = int(S_max)
    S_range = np.arange(S_min,S_max,0.1)
    calls = [CallPrice(S,K,T,r,sigma) for S in S_range] # Establish y-values for call/put options
    puts = [PutPrice(S,K,T,r,sigma) for S in S_range]
    plt.plot(S_range,calls,label="Call Value") # Visualize call-put parity
    plt.plot(S_range,puts,label="Put Value")
    plt.xlabel("$S_0$")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
  elif sensitivity_choice == "N": # Data validity check
    print("Call-Put Parity graph not selected.")
  else:
    print("Invalid data input - please restart code.")

# Tool 3
def tool_3():
  from getpass import getpass
  API_KEY = getpass("Please input your AlphaVantage API KEY: ")

  import pandas as pd # Import packages for math/data visualization
  import numpy as np
  import matplotlib.pyplot as plt
  # Import the CSV Data from AlphaVantage
  symbol_list = [] # Establish variables as empty lists & dataframes
  stock_list =[]
  df_list = []
  df_p_list = []
  df_p = pd.DataFrame()
  returns_p = pd.DataFrame()

  while True: # Allow for multiple stock inputs with while loop, exit loop once all stocks have been inputted
    symbol_input = input("Please input the ticker symbol for your stock: ")
    if symbol_input == "DONE":
      break
    else:
      symbol_input = symbol_input.upper()
      symbol_list.append(symbol_input)
  for x in range(len(symbol_list)): # Create dataframe of portfolio & returns using correct stocks
    stock_data = pd.read_csv(f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol_list[x]}&apikey={API_KEY}&datatype=csv')
    stock_list.append(stock_data)
    df_data = pd.DataFrame(stock_data)
    df_list.append(df_data)
    df_p[symbol_list[x]] = df_data['adjusted_close']
    df_p_list.append(df_p)
    returns_p[symbol_list[x]] = df_p[symbol_list[x]].pct_change()

  print("STOCKS:",symbol_list,) # Output the returns portfolio with listed stocks
  print(returns_p)

  cov_matrix = returns_p.cov()*252 # Calculate and display the covariance matrix
  print("COVARIANCE MATRIX:")
  cov_matrix

  p_returns = [] # Create more variables for efficient frontier determination
  p_volatility = []
  p_weights = []
  assets = len(df_p.columns)
  portfolios = 10000
  i_returns =  returns_p.mean()

  for x in range(portfolios): # Simulation model to generate possible portfolios
    weights = np.random.random(assets)
    weights = weights / np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, i_returns)
    p_returns.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights,axis=1).sum().sum()
    sd = np.sqrt(var)
    ann_sd = sd*np.sqrt(250)
    p_volatility.append(ann_sd)
  
  data = {'Returns':p_returns, 'Volatility':p_volatility} # Create weighted portfolio dataframe with all simulations
  for a, b in enumerate(df_p.columns.tolist()):
    data[b+' weight'] = [w[a] for w in p_weights]
  final_df = pd.DataFrame(data)
  print(final_df.head()) # Display dataframe and covariance matrix
  print(cov_matrix)

  final_df.plot.scatter(x='Volatility',y='Returns') # Visualize the efficient frontier
  plt.xlabel("Risk")
  plt.ylabel("Expected Returns")

  min_vol = final_df.iloc[final_df['Volatility'].idxmin()] # Determine portfolio with the lowest volatility and output
  print("Minimum Volatility Portfolio:")
  print(min_vol)

  rf = input("Please input risk tolerance") # Determine optimal risky portfolio given risk tolerance value and output
  rf = float(rf)
  optimal_p = final_df.iloc[((final_df['Returns']-rf)/final_df['Volatility']).idxmax()]
  print("Optimal Risky Portfolio:")
  print(optimal_p)
      

# Decision Support System:
import os
print("Hello!")
print("1 - Stock Price Data Visualization")
print("2 - Option Pricing/Delta Hedging Tool")
print("3 - Portfolio Optimization")
API_KEY = os.getenv("API_KEY")
tool_number = input("Please select which tool you would like to use (1-3): ")
if tool_number == "1": # Call requested functionality based on user input
  tool_1()
elif tool_number == "2":
  tool_2()
elif tool_number == "3":
  tool_3()
else:
  print("Invalid tool number") # Validity check for incorrect input values