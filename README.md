# Finance DSS
## Setup


Create and activate a virtual environment:


```sh 
conda create -n kiwi-env python=3.8

conda activate kiwi-env
```

Install package dependencies:

```sh
pip install -r requirements.txt
```

## Configuration

```
The local .env file containing the confidential API_KEY has already been implemented, so there is no need for further configuration.
```


## Usage

Run an example script:

```sh
python -m app.alpha
```

Run the decision supprt system:

```sh
#python app/dss.py

python -m app.dss
```

## Help Guide

To use the decision support system, please select the desired application tool by entering the designated number:
* 1 - Stock Price Data Visualization
* 2 - Option Pricing Tool
* 3 - Portfolio Optimization Tool

### Tool 1: Stock Price Data Visualization
To visualize the stock price over the past 100 days, please enter the ticker symbol for your desired stock.

### Tool 2: Option Pricing Tool
To determine the values of call/put options at a strike price, please enter the ticker symbol for your desired stock and the strike price of your asset.
* If you wish to view the Greeks (financial parameters) for this option, please type "Y" when requested.
* If you wish to view the call-put parity graph for this option, please type "Y" when requested.

### Tool 3: Portfolio Optimization Tool
To view the efficient frontier minimum volatility portfolio for a given set of assets, please enter the ticker symbol of each stock in your prospective portfolio. When finished, please type "DONE" when requested.
* To view the optimal risky portfolio, please input your risk tolerance (A) when requested.
* NOTE: Risk tolerance generally varies from 0 to 0.1, with 0.05 being the industry standard. A lower value implies risk-aversion, while a higher value implies risk-taking behavior in investing.
