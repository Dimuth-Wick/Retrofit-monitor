"""Train and compare the Retrofit Monitor models on the fake feature table.

THE BIG PICTURE (read this first)
---------------------------------
We have a table. Each row = one 1-second window of the machine.
Columns = features (numbers that describe the window) + the true state.

    rms_current_a  peak_current_a  crest_factor  vib_rms_g  vib_dom_freq_hz | state
    1.80           2.67            1.44          0.10       6.0             | Wash

Machine learning = find a rule that goes from the five feature numbers to the state.

TWO SEPARATE JOBS in this file, using DIFFERENT kinds of algorithm:

  PART 1  State classification  (SUPERVISED: we have the true labels)
          Four algorithms compete: KNN, Logistic regression,
          Decision tree, Random forest.   Same data, same split, fair fight.

  PART 2  Fault detection       (UNSUPERVISED anomaly detection)
          One algorithm: Isolation Forest.
          It sees ONLY normal windows and learns "what normal looks like".
          Anything that looks unusual is flagged as a possible fault.
          (Why not just classify "Fault" like the other states? Real faults are
          rare, so in real life you will almost never have enough fault examples.)

Run first:  python make_fake_dataset.py
Then:       python Classifier_model.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                       # draw plots into files, no pop-up windows
import matplotlib.pyplot as plt
import joblib                               # saves a trained model to a file
from pathlib import Path

# scikit-learn gives us ready-made algorithms. We import only the pieces we use.
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             ConfusionMatrixDisplay, classification_report)

# ---------------- Settings ----------------
# The columns the model is allowed to look at. NOTE: "state" is NOT in this list.
# The state is the ANSWER; the model must work it out from these five numbers.
FEATURES = ["rms_current_a", "peak_current_a", "crest_factor",
            "vib_rms_g", "vib_dom_freq_hz"]
TEST_FRACTION = 0.2       # keep 20% of the cycles hidden, for the final exam
CONTAMINATION = 0.019       # fault detector: allow ~2% false alarms on normal data
SEED = 42                  # fixed seed = same random choices every run

folder = Path(__file__).parent              # the folder this script is in
rng = np.random.default_rng(SEED)

# ---------------- 1. Load the data ----------------
df = pd.read_csv(folder / "data" / "fake_features.csv")

# Clean the text. Some editors "align" CSV columns by adding spaces, which turns
# the column name "fault_type" into " fault_type" and the value "Idle" into
# "Idle ". Python then sees them as different words and errors or mislabels.
# .strip() removes spaces at the start and end.
df.columns = df.columns.str.strip()
for col in ["state", "fault_type"]:
    df[col] = df[col].str.strip()
print(f"Loaded {len(df)} windows from {df['cycle_id'].nunique()} cycles\n")

# ---------------- 2. Split into TRAIN and TEST ----------------
# Idea: the model learns from the TRAIN part, then we examine it on the TEST part
# that it has never seen. This is your 80/20 split from the notes. The test score
# tells us how it would do on NEW data. (Score on the training data would be
# useless: that is the overfitting trap, "brilliant on training, terrible on testing".)
#
# BUT we split by whole CYCLE, not by random window. Reason: windows from the same
# cycle look alike (same load of clothes, same machine mood). If window 5 of a
# cycle were in training and window 6 in test, the test would be secretly
# "copying" the training data and the score would be fake-good. This mistake is
# called DATA LEAKAGE.
#
# Fault cycles and normal cycles are split separately, so BOTH reach the test set.
cycles = df.groupby("cycle_id")["fault_type"].first()   # one label per cycle: none/overload/unbalanced
test_cycles = []
for is_fault_cycle in (False, True):
    ids = cycles[(cycles != "none") == is_fault_cycle].index.to_numpy()
    n_test = max(1, round(len(ids) * TEST_FRACTION))
    test_cycles += list(rng.choice(ids, size=n_test, replace=False))   # pick random cycles

is_test = df["cycle_id"].isin(test_cycles)    # True for rows that belong to a test cycle
train, test = df[~is_test], df[is_test]       # ~ means "not"
print(f"Train: {train['cycle_id'].nunique()} cycles ({len(train)} windows)   "
      f"Test: {test['cycle_id'].nunique()} cycles ({len(test)} windows)\n")

# ======================================================================
# PART 1: STATE CLASSIFICATION (supervised)
# ======================================================================
print("=" * 60)
print("PART 1  State classification")
print("=" * 60)

# This part only handles the 5 normal states, so drop the Fault windows.
train_s = train[train["state"] != "Fault"]
test_s = test[test["state"] != "Fault"]

X_train, y_train = train_s[FEATURES], train_s["state"]
X_test, y_test = test_s[FEATURES], test_s["state"]

# The four competitors. Short idea of each (you already studied these):
#   KNN                : look at the 5 most similar training windows (nearest
#                        neighbours by distance), take a vote.
#   Logistic regression: a weighted sum of the features, turned into a probability
#                        for each state (the sigmoid / softmax from your notes).
#   Decision tree      : a chain of yes/no questions, e.g. "vib_dom_freq_hz > 15?",
#                        choosing each question to reduce Gini impurity the most.
#   Random forest      : 100 decision trees, each trained on a slightly different
#                        sample of the data, that vote together.
#
# make_pipeline(StandardScaler(), ...) = "first rescale every feature to a similar
# size, THEN run the algorithm". Why: KNN measures distance, and a feature like
# vib_dom_freq_hz (about 18) would drown a feature like vib_rms_g (about 0.1).
# Trees ask "is this number above X?" for one feature at a time, so they do not
# care about size and need no scaler.
models = {
    "KNN":                 make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Decision tree":       DecisionTreeClassifier(max_depth=6, random_state=SEED),   # max_depth limits overfitting
    "Random forest":       RandomForestClassifier(n_estimators=100, random_state=SEED),
}

scores = {}
for name, model in models.items():
    model.fit(X_train, y_train)               # <- THIS LINE IS THE TRAINING. The model studies the rows.
    y_guess = model.predict(X_test)           # now it guesses the states of the hidden test rows
    scores[name] = accuracy_score(y_test, y_guess)   # accuracy = fraction of guesses that are correct
    print(f"{name:<22} test accuracy = {scores[name]:.3f}")

best_name = max(scores, key=scores.get)       # the name with the highest score
best = models[best_name]
print(f"\nBest on this split: {best_name}")
# Note: the scores are all close. On one small split, a 0.5% gap is luck, not
# proof. Cross-validation (a later topic) is the proper way to compare.

# Accuracy is ONE number. It does not say WHICH states get mixed up.
# The confusion matrix does: rows = the true state, columns = what the model said.
# A perfect model has numbers only on the diagonal.
labels = ["Idle", "Wash", "Rinse", "Drain", "Spin"]
y_pred = best.predict(X_test)
print("\nPer-state results")
print("  precision = when it says 'Wash', how often is it really Wash?")
print("  recall    = of all the real Wash windows, how many did it find?")
print(classification_report(y_test, y_pred, labels=labels, digits=3))

(folder / "results").mkdir(exist_ok=True)
cm = confusion_matrix(y_test, y_pred, labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
plt.title(f"State classification: {best_name}")
plt.savefig(folder / "results" / "confusion_matrix_state.png", dpi=150, bbox_inches="tight")
plt.close()

# Which features did the random forest rely on most? (Useful: tells you which
# sensor matters, and which feature you might drop to save ESP32 memory.)
rf = models["Random forest"]
imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
imp.plot(kind="barh", title="Random forest: feature importance")
plt.savefig(folder / "results" / "feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# ======================================================================
# PART 2: FAULT DETECTION (anomaly detection)
# ======================================================================
print("=" * 60)
print("PART 2  Fault detection")
print("=" * 60)

# Isolation Forest idea (no maths needed): pick a random feature and a random
# cut value, and keep cutting the data into smaller groups. A strange window
# sits far from the crowd, so it gets ALONE after only a few cuts. A normal
# window sits inside the crowd and needs many cuts. "Few cuts to isolate" = unusual.
# It is related to your clustering notes: "far from the normal group" = anomaly.
#
# Train on normal windows ONLY. The detector never sees a single fault.
normal_train = train[train["state"] != "Fault"][FEATURES]
detector = IsolationForest(n_estimators=200,              # 200 random trees
                           contamination=CONTAMINATION,   # the strictness knob (see below)
                           random_state=SEED)
detector.fit(normal_train)

# Now test on the hidden cycles, which DO contain faults.
X_t = test[FEATURES]
is_fault = (test["state"] == "Fault").to_numpy()     # the true answer (only used for scoring)
flagged = detector.predict(X_t) == -1                # the detector says -1 = "unusual", 1 = "normal"

# Two numbers matter, and they pull in opposite directions:
#   detected    = of the REAL faults, what fraction did it flag?      (spec PC2: at least 90%)
#   false_alarm = of the NORMAL windows, what fraction did it flag?   (spec PC3: below 5%)
# CONTAMINATION is the knob: raise it and you catch more faults but also raise
# more false alarms. You are choosing where to sit on that trade-off.
detected = flagged[is_fault].mean()
false_alarm = flagged[~is_fault].mean()
print(f"Faults detected : {detected:.1%}   (spec target: at least 90%)")
print(f"False alarms    : {false_alarm:.1%}   (spec target: below 5%)")

# Break the result down by fault type. Overload is easy (huge current); the
# unbalanced load is harder because it hides inside normal-looking Spin values.
for ftype in ("overload", "unbalanced"):
    mask = (test["fault_type"] == ftype).to_numpy() & is_fault
    if mask.any():
        print(f"   {ftype:<11} faults detected: {flagged[mask].mean():.1%}  ({mask.sum()} windows)")

# WHY accuracy is a trap when faults are rare:
share_normal = (~is_fault).mean()
print(f"\nA useless model that always says 'normal' would score {share_normal:.1%} accuracy")
print("while detecting 0% of faults. That is why we look at detection rate and false alarms.")

# ===== PART 3: FAULT TYPE (percentages) =====
fault_train = train[train["state"] == "Fault"]
fault_test = test[test["state"] == "Fault"]

type_model = RandomForestClassifier(n_estimators=100, random_state=SEED)
type_model.fit(fault_train[FEATURES], fault_train["fault_type"])

type_acc = accuracy_score(fault_test["fault_type"], type_model.predict(fault_test[FEATURES]))
print(f"Fault-type accuracy = {type_acc:.3f}")

proba = type_model.predict_proba(fault_test[FEATURES])
for true, p in list(zip(fault_test["fault_type"], proba))[:6]:
    print(true, "→", "  ".join(f"{c}={x:.0%}" for c, x in zip(type_model.classes_, p)))

# ---------------- Save the trained models to files ----------------
# Once saved, you can load a model later without training again. (A saved model
# can break if the scikit-learn version changes, which is one reason to keep the
# versions fixed in a requirements.txt file.)
(folder / "models").mkdir(exist_ok=True)
joblib.dump(models["Random forest"], folder / "models" / "state_random_forest.joblib")
joblib.dump(detector, folder / "models" / "fault_isolation_forest.joblib")
joblib.dump(type_model, folder / "models" / "fault_type_forest.joblib")

print("\nSaved models to ml/models/ and plots to ml/results/")
