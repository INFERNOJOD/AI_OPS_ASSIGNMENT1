**Question 1 — Technical Debt Diagnosis**

**1. Hidden technical debt categories**

* **(a) Entanglement (CACE - Changing Anything Changes Everything):** Delivery-time and recommendation features are tightly coupled, so tweaking one silently broke the other, even though they seem unrelated.
* **(b) Undeclared consumers:** Marketing dashboard is quietly reading the model's output table with no contract, so the ML team can't change the output without maybe breaking their dashboard.
* **(c) Pipeline jungles:** The training pipeline consists of 14 undocumented shell scripts with no orchestration, creating an unmanageable and unreproducible tangle of glue code.

**2. Mitigation**

* **For (c) Pipeline jungles:** Use Apache Airflow to turn the scripts into a proper DAG with defined dependencies, retries, and logging instead of blind shell chaining.
