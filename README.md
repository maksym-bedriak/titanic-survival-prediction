# Titanic Survival Prediction

Predicting passenger survival on the Titanic using classic ML techniques, built as a hands-on introduction to the ML workflow.

## Overview
- Dataset: Kaggle Titanic competition
- Goal: Predict whether a passenger survived based on features like class, sex, age, and family size
- Models: Logistic Regression, Random Forest

## Approach
1. Dropped uninformative columns (PassengerId, Name, Ticket, Cabin)
2. Filled missing values in Age (median) and Embarked (mode)
3. Encoded Sex as binary, one-hot encoded Embarked
4. Engineered FamilySize (SibSp + Parch + 1) and AgeBin (binned age groups), then dropped the original SibSp/Parch columns
5. Compared Logistic Regression vs Random Forest
6. Tuned Random Forest (max_depth, min_samples_leaf, min_samples_split) to fix an initial overfitting issue (98% train / 77% test accuracy on an untuned model)
7. Validated the tuned model with 5-fold cross-validation

## Results
| Model | Test Accuracy |
|---|---|
| Logistic Regression | ~76% |
| Random Forest (tuned) | ~78% |

5-fold CV mean accuracy on the tuned Random Forest: **~82%** (std ≈ 0.03), suggesting the single test-split score understates real performance.

## Key Insight
Sex and Passenger Class were the strongest predictors of survival in Logistic Regression, consistent with the historical "women and children first" evacuation priority.

Random Forest told a slightly different story: `Fare` ranked above `Pclass` in feature importance, even though Logistic Regression barely weighted it. This makes sense — Fare is a more granular, continuous proxy for the same wealth/class signal that Pclass captures only in three discrete buckets. Random Forest's tree splits can exploit that extra resolution, while Logistic Regression's single linear coefficient cannot. A useful reminder that "feature importance" isn't a fixed property of the data — it depends on what the model is capable of extracting.

An untuned Random Forest also overfit badly; constraining tree depth and leaf size closed most of the train/test gap.

## How to Run
\`\`\`bash
pip install -r requirements.txt
jupyter notebook titanic.ipynb
\`\`\`

## Tech Stack
Python, pandas, scikit-learn, matplotlib