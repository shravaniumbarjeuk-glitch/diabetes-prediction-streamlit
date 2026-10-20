# Diabetes Prediction App (Streamlit)

A small web app that predicts whether a person is likely to be diabetic from eight clinical measurements, using a **linear Support Vector Machine** trained on the Pima Indians Diabetes dataset.

Created for the honours mini-project *Predicting Diabetes Onset: A Machine Learning Approach* (Walchand Institute of Technology, 2023-24), a team project whose technology stack was pandas, NumPy, scikit-learn and Streamlit. It follows a standard introductory workflow (standardise, split, train a linear SVM, wrap it in a form) and is a deliberately simple baseline, not a clinical tool.


## What it does

1. Loads `diabetes.csv` and standardises the eight features (`StandardScaler`).
2. Splits the data 80/20, stratified by outcome (`random_state=2`).
3. Trains an `SVC(kernel='linear')` classifier.
4. Shows a **Streamlit** form in the sidebar for the eight inputs and returns "diabetic" or "not diabetic" for the entered values.

| Input | Meaning |
|---|---|
| Pregnancies | number of pregnancies |
| Glucose | plasma glucose concentration |
| BloodPressure | diastolic blood pressure (mm Hg) |
| SkinThickness | triceps skin-fold thickness (mm) |
| Insulin | 2-hour serum insulin |
| BMI | body mass index |
| DiabetesPedigreeFunction | diabetes pedigree score |
| Age | years |

## Run it

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (on macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
# place diabetes.csv in this folder (see data/README.md)
streamlit run app.py
```

## Model performance

The app itself does not display accuracy. I measured it on the public Pima Indians Diabetes data (768 rows: 500 not diabetic, 268 diabetic) using exactly the settings in `app.py` (80/20 stratified split, `random_state=2`, linear SVM, 614 training and 154 test rows):

| Measure | Result |
|---|---|
| **Test accuracy** | **77.3%** (119 of 154 correct) |
| Training accuracy | 78.7% |
| Test accuracy with the scaler fitted on the training split only | 77.3% |
| 5-fold stratified cross-validation (scaler inside each fold) | 77.7% (standard deviation 2.9 points) |
| Majority-class baseline (always predict "not diabetic") | 65.1% |
| Precision for "diabetic" | 75.7% |
| **Recall for "diabetic"** | **51.9%** (28 of 54 diabetic cases found) |

The model beats the baseline by about 12 points, but it misses almost half of the diabetic cases, so accuracy alone flatters it. Test confusion matrix: 91 true negatives, 9 false positives, 26 false negatives, 28 true positives.

To reproduce, run the app's code with `diabetes.csv` present and add `print(accuracy_score(Y_test, classifier.predict(X_test)))` after training.

## Limitations

- A baseline only: no hyper-parameter tuning, class-imbalance handling or calibration, and a small test set (154 rows).
- The scaler is fitted on the whole dataset before the split. This did not change the result here (see the table), but fitting it on the training split only is the cleaner practice.
- Recall for diabetic cases is only about 52%, so the app would miss many diabetic people.
- The dataset is small and describes a specific population, so results do not generalise.
- In the dataset, some zero values (for example blood pressure or insulin) stand for missing measurements and are not treated specially here.
- **This is an educational project and must not be used for medical decisions.**

## Data

The Pima Indians Diabetes dataset is not stored here. See [`data/README.md`](data/README.md).

## Tools

Python, pandas, NumPy, scikit-learn, Streamlit.

## Author

Shravani Umbarje. MIT licence (see `LICENSE`).
