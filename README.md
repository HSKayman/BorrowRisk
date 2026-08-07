# BorrowRisk — Loan Default Prediction

Data-science capstone project that predicts whether a borrower will default on a loan, helping financial institutions assess lending risk.

## Objective

Build and compare classifiers that decide if a borrower is likely to repay or default. The workflow covers data preparation, class-imbalance strategies, model training/evaluation, and a lightweight Streamlit deployment for interactive scoring.

## Repository Contents

| Path | Description |
|------|-------------|
| `HSK_Capstone_v5.ipynb` | Full notebook: EDA, preprocessing, imbalance handling (SMOTE / ADASYN / undersampling), and model comparison |
| `HSK_Capstone_v5.py` | Script export of the same capstone pipeline |
| `deployment.py` | Streamlit UI that loads the trained Random Forest + scaler and scores new loan applications |
| `Loan_Default.csv` | Training/evaluation dataset |
| `random_forrest_final.pkl` | Serialized Random Forest model |
| `standart_scaler_final.plk` | Serialized feature scaler used at inference |

## Pipeline Highlights

1. **Preprocessing** — cleaning, missing values, outlier analysis  
2. **Imbalance strategies** — SMOTE, ADASYN, undersampling  
3. **Models evaluated** — KNN, Logistic Regression, Neural Nets, Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost  
4. **Deployment** — Streamlit app (`deployment.py`) for interactive predictions  

## Quick Start (deployment)

```bash
pip install streamlit pandas scikit-learn
streamlit run deployment.py
```

Ensure `random_forrest_final.pkl` and `standart_scaler_final.plk` are in the working directory.

## License

See [LICENSE](LICENSE).
