# Assignment 1: MLflow and DVC 

**Author:** Krish Yadav (DA24B043)  
**Demo Video:** [Insert Google Drive/YouTube Link Here - Ensure access is public]  
**Q4 Collaborative Repository:** [Insert Q4 Repo Link Here, if kept separate]  

This repository contains the required code files, MLflow tracking scripts, and DVC configurations for Assignment 1. 

**Repository Structure**
* `Q1/`: Contains scripts for basic MLflow experiment tracking, parameter logging, and metric recording. 
* `Q2/`: Contains scripts demonstrating model registry operations and hyperparameter tuning tracked via MLflow.
* `Q3/`: Contains the DVC pipeline demonstrating dataset versioning, updates, and rollback operations configured with an external SSH remote.
* `Q4/`: Contains the collaborative MLflow workflow. Commits from both partners are strictly distinguishable.

**Setup & Execution Instructions**
1. **Prerequisites:** Ensure `mlflow` and `dvc[ssh]` are installed in your environment.
2. **Q1 & Q2:** Navigate to the respective directories and run the Python scripts. Execute `mlflow ui` in the terminal to view the logged runs on `localhost:5000`.
3. **Q3:** The DVC remote is configured for an external SSH server. To view the local state, you can inspect `.dvc/config` and run `dvc status`. 
4. **Q4:** Check the Git commit history to verify distinguishable contributions from both partners.
