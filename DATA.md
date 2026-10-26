# Data

**Credit Card Fraud Detection (ULB, Dal Pozz et al. 2015)**

- Rows: 284,807 transactions, 31 columns
- Source: public Kaggle/ULB mirror David-Egea/Creditcard-fraud-detection, complete file
- Licence: research use; features already PCA-anonymised by the data owners
- Class balance is 0.173 percent, matching the published figure. V1 to V28 are anonymised principal components, so no feature has a business-readable name.

Raw files are **not committed**. Drop the file into `data/` under the name the
script expects, or point a symlink there, then run `python src/main.py`.
