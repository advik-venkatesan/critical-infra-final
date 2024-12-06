// SPDX-License-Identifier: MIT
pragma solidity 0.8.19;

contract SecureContract {
    mapping(address => uint256) public balances;

    // Deposit function to add Ether to the contract
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // Withdraw function with updated security
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        balances[msg.sender] -= amount; // Update state before external call
        payable(msg.sender).transfer(amount); // Safe transfer to avoid low-level call issues
    }
}
