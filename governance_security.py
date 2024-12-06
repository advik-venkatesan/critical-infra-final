import numpy as np

def calculate_gini_coefficient(stakes):
    """Calculate Gini coefficient for stake distribution."""
    sorted_stakes = np.sort(stakes)
    n = len(stakes)
    cumulative_stakes = np.cumsum(sorted_stakes)
    return (2 * np.sum((np.arange(1, n + 1) * sorted_stakes)) / cumulative_stakes[-1] - (n + 1)) / n


def simulate_voting_attack(voting_data, attacker_stake):
    """Simulate a voting attack by adding an attacker's stake."""
    max_voter = voting_data['stake'].idxmax()
    voting_data.loc[max_voter, 'stake'] += attacker_stake
    return calculate_gini_coefficient(voting_data['stake'])


def load_mock_dao_data():
    """Simulated DAO governance data."""
    import pandas as pd
    # The first one is good balanced data, and the other centralized governance
    return pd.DataFrame({
        'proposal_id': [1, 1, 1, 1],
        'voter_address': ['0xvoter1', '0xvoter2', '0xvoter3', '0xvoter4'],
        'stake': [500, 300, 200, 500]
    }), pd.DataFrame({
        'proposal_id': [1, 1, 1, 1],
        'voter_address': ['0xvoter1', '0xvoter2', '0xvoter3', '0xvoter4'],
        'stake': [500, 300, 200, 10000]
    })
