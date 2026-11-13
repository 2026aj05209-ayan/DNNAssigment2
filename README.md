# Deep Neural Networks Assignment 2: Recurrent Networks & Transformers

**Student Name:** Ayan Chawla
**Student ID:** 2026AJ05209  
**Variant:** [Task 5(c) Ablation: Gradient Clipping (Even Digit) / Forget-Gate Bias (Odd Digit)]

---

## 1. Environment & Dependencies

- **Python Version:** 3.10+
- **PyTorch Version:** 2.x
- **Libraries:** NumPy, pandas, Matplotlib, seaborn, scikit-learn
- **Hardware Platform:** Google Colab T4 GPU

All required packages are listed in `requirements.txt`.

---

## 2. Execution Instructions

Both notebooks use relative paths and automatically handle dataset downloads.

1. **Problem 1 (`problem1/problem1.ipynb`)**:
   - Open in Jupyter or Colab and select **Restart and Run All**.
   - Downloads UCI HAR, runs unit tests for custom cells (`MyRNNCell`, `MyLSTMCell`, `MyGRUCell`), executes classification and adding problem benchmarks, saves figures to `problem1/figures/`, and appends metrics to `results.csv`.

2. **Problem 2 (`problem2/problem2.ipynb`)**:
   - Open in Jupyter or Colab and select **Restart and Run All**.
   - Runs attention/encoder unit tests, trains Task A (HAR Encoder Classifier) and Task B (Tiny Shakespeare Causal LM), generates text samples in `generated_samples.txt`, executes analytical ablations/heatmaps, saves plots to `problem2/figures/`, and logs metrics to `results.csv`.

---

## 3. Total Run Time (T4 GPU)

- **Problem 1 Execution Time:** ~[X] minutes
- **Problem 2 Execution Time:** ~[X] minutes (Task B Causal LM: ~10–15 minutes)
- **Total Pipeline Execution Time:** ~[X] minutes

---

## 4. Assumptions & Known Issues

- Fixed random seeds (`seed=42, 43, 44`) ensure reproducibility across all runs.
- [Mention any assumptions made or specific CPU/GPU configurations used].
