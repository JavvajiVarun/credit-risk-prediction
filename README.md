cat > README.md << 'EOF'
# Cost-Sensitive & Fairness-Aware Credit Risk Prediction

## What this project does
Predicts loan defaults using the Give Me Some Credit dataset (~150K records).
Goes beyond standard accuracy metrics by incorporating:
- Expected Loss optimization (loan amount x LGD)
- Fairness analysis across age groups
- Per-group threshold calibration

## Pipeline
1. Data loading & preprocessing
2. Random Forest with sample reweighting
3. Expected Loss threshold sweep (0.01 to 0.99)
4. Fairness fix — per-group thresholds
5. Full visualization

## Key Results
- ROC-AUC: 0.87 (real dataset)
- Default recall improved: 54% to 87%
- Expected Loss reduced by 5.93M vs default threshold
- TPR fairness gap reduced: 55% to 32%

## How to run
pip install scikit-learn matplotlib seaborn pandas numpy
python complete_pipeline.py

## Dataset
Download cs-training.csv from:
https://www.kaggle.com/competitions/GiveMeSomeCredit/data
Place in ./data/ folder.
EOF
