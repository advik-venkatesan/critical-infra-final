from governance_security import calculate_gini_coefficient, load_mock_dao_data
from oracle_security import detect_price_anomalies, fetch_mock_oracle_prices
from contract_security import check_contract_vulnerabilities

def run_protect_framework_simulated():
    # Governance Security
    print("\n=== Analyzing DAO Governance Security ===")
    good_dao_data, bad_dao_data = load_mock_dao_data()
    print("Good DAO Governance:")
    gini_coeff = calculate_gini_coefficient(good_dao_data['stake'])
    print(f"DAO Gini Coefficient: {gini_coeff}")
    print("\nBad DAO Governance:")
    gini_coeff = calculate_gini_coefficient(bad_dao_data['stake'])
    print(f"DAO Gini Coefficient: {gini_coeff}")

    # Oracle Security
    print("\n=== Monitoring Oracle Security ===")
    good_oracle_prices, bad_oracle_prices = fetch_mock_oracle_prices()
    print("Good Oracle Data:")
    good_anomalies = detect_price_anomalies(good_oracle_prices)
    print("Anomalies:", good_anomalies)
    print("\nBad Oracle Data:")
    bad_anomalies = detect_price_anomalies(bad_oracle_prices)
    print("Anomalies:", bad_anomalies)

    # Contract Security
    print("\n=== Contract Security ===")
    print("\nBad Contract:")
    bad_contract_path = "contracts/bad_contract.sol" 
    contract_status, _ = check_contract_vulnerabilities(bad_contract_path)
    print(f"Contract Analysis Result: {contract_status}")
    print("\nGood Contract:")
    good_contract_path = "contracts/good_contract.sol" 
    contract_status, _ = check_contract_vulnerabilities(good_contract_path)
    print(f"Contract Analysis Result: {contract_status}")

if __name__ == "__main__":
    run_protect_framework_simulated()
