# Clinical Outcomes Analysis Pipeline

Pipeline for clinical outcomes (e.g. mortality / readmission). Designed for **MIMIC** (with official access); methodology applies to similar clinical DBs.

---

## Project structure

```
MIMIC-Analysis_Pipeline-main/
├── MIMIC_Analysis.py      # entry: MIMICPipeline
├── content/
│   ├── preprocessing_methods.py   # impute, scale, one-hot
│   ├── feature_selection.py      # MI, RFE, SMD tables
│   ├── model_training.py          # LR, XGB, RF, AdaBoost, NB, SVM, LightGBM
│   ├── model_explainers.py        # SHAP, LIME, ablation
│   └── functions.py               # ROC/calibration plots, threshold metrics
├── datasets/               # put Excel/CSV here
├── outputs/                # ROC, calibration, DCA PNGs; threshold CSV (created on run)
├── requirements.txt
└── README.md
```

---

## Run

```bash
pip install -r requirements.txt
python MIMIC_Analysis.py
```

Data: put your Excel file under `datasets/` and set `path` in `MIMIC_Analysis.py` (e.g. `path="./datasets/saaki.xlsx"`).  
`y_position`: column index of the target (0-based).  
Results: `outputs/` (ROC, calibration, decision curve analysis; `threshold_metrics_*.csv`).

---

## Related publication

- [Predicting ICU Readmission in Acute Pancreatitis Patients Using a Machine Learning-Based Model with Enhanced Clinical Interpretability](https://arxiv.org/abs/2505.14850)

---

## Notes

- Research / academic use. Comply with data use agreements when using other datasets.
