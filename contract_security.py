def format_slither_report(report, contract_path):
    """Format the raw Slither report into a structured output."""
    formatted_report = []
    
    formatted_report.append(f"#### **Contract Path:** `{contract_path}`\n")
    vulnerabilities = []
    lines = report.splitlines()
    for i, line in enumerate(lines):
        if "Reentrancy" in line or "External calls" in line or "State variables" in line:
            vulnerabilities.append("\n".join(lines[i:i+10]))
    
    if vulnerabilities:
        formatted_report.append("---")
        formatted_report.append("#### **Analysis Result:**")
        formatted_report.append("**Error**\n")
        formatted_report.append("---")
        formatted_report.append("### **Vulnerabilities Found**")
        
        for i, vuln in enumerate(vulnerabilities, 1):
            formatted_report.append(f"#### **{i}.** {vuln}")
        
        formatted_report.append("---")
    else:
        formatted_report.append("#### **Analysis Result:**")
        formatted_report.append("**Good Contract**\n")
        formatted_report.append("No Vulnerabilities\n")
        formatted_report.append("---")
    
    return "\n".join(formatted_report)

import subprocess

def check_contract_vulnerabilities(contract_path):
    """Run Slither analysis on the provided Solidity contract."""
    try:
        print("Starting Slither analysis...") 
        result = subprocess.run(
            ['slither', contract_path, '--exclude-informational'],
            capture_output=True,
            text=True
        )
        print("Slither analysis completed.")

        if result.returncode == 0:
            report = result.stdout
            if "Detectors" in report or "Vulnerabilities" in report:
                formatted_report = format_slither_report(report, contract_path)
                return "Bad Contract", formatted_report
            else:
                print(f"Slither Analysis Report:\n\nContract Analysis Result: Good Contract\nNo Vulnerabilities\n")
                return "Good Contract", None
        else:
            print(f"Error during Slither analysis:\n{result.stderr}")
            return "Error", result.stderr
    except Exception as e:
        print(f"Error running Slither: {e}")
        return "Error", str(e)
