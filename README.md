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


[Obtain an API Key](https://www.alphavantage.co/support/#api-key) from AlphaVantage. For the stocks report to work, you'll need a "premium" key.

Then create a local ".env" file and provide the key like this:

```sh
# this is the ".env" file...

ALPHAVANTAGE_API_KEY="_________"
```


## Usage

Run an example script:

```sh
python -m app.alpha
```

Run the decision supprt system:

```sh
#python app/unemployment.py

python -m app.dss
```
