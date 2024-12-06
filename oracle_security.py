import numpy as np

def detect_price_anomalies(prices):
    """Detect anomalies in price data."""
    mean_price = np.mean(list(prices.values()))
    anomalies = {oracle: price for oracle, price in prices.items() if abs(price - mean_price) > 0.05 * mean_price}
    return anomalies


def analyze_historical_oracle_data(historical_data):
    """Analyze historical oracle data for anomalies."""
    historical_data['deviation'] = abs(historical_data['price'] - historical_data['expected_price']) / historical_data['expected_price']
    anomalies = historical_data[historical_data['deviation'] > 0.05]
    return anomalies


def fetch_mock_oracle_prices():
    """Simulated oracle price feeds."""
    # The first is consistent, good data, and the second one includes and an outlier
    return {
        '0xOracleAddress1': 100,
        '0xOracleAddress2': 95,
        '0xOracleAddress3': 105,
        '0xOracleAddress4': 100
    }, {
        '0xOracleAddress1': 100,
        '0xOracleAddress2': 95,
        '0xOracleAddress3': 110,
        '0xOracleAddress4': 150
    }
