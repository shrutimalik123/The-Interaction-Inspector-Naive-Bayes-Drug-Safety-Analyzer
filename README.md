# 🩺 The Interaction Inspector: Naive Bayes Drug Safety Analyzer

An interactive, probabilistic Supervised Learning simulation designed to teach **Naive Bayes Classification** and **Bayes' Theorem** from scratch. You play as a Clinical Informatics Pharmacist coding an automated Clinical Decision Support System (CDSS) that cross-examines drug therapeutic profiles and patient metabolic rates to prevent life-threatening Drug-Drug Interactions (DDIs).

## 🎓 Learning Objectives

This project focuses on teaching:
* **Naive Bayes Classifiers:** A foundational classification algorithm anchored on computing continuous probability statistics using conditional event metrics.
* **The "Naive" Independence Assumption:** Understanding the massive mathematical shortcut of treating multi-dimensional variables as entirely independent factors to maximize calculation speed.
* **Prior vs. Posterior Probabilities:** Tracking how a baseline risk profile ($P(H)$) updates dynamically into a specialized risk verdict ($P(H|E)$) once unique patient evidence is introduced.
* **The Zero-Probability Trap:** Identifying how unseen categorical values can introduce absolute zeroes that collapse an entire multiplication pipeline (and how to fix it).

---

## ✨ Features

* **Clinical Decision Support Scenario:** Applies conditional probability concepts to an enterprise health-tech framework tracking patient treatment safety.
* **Mathematical Transparency Matrix:** Breaks down and prints the exact historical priors, feature likelihoods, and relative posterior outcomes step-by-step.
* **Pure Logical Branching:** Built completely inside standard Python, calculating compound Bayesian probabilities natively without importing external machine learning data structures.

---

## 🚀 How to Run the Game

### 1. Prerequisites
You only need **Python 3** installed.

### 2. Setup and Execution
1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/interaction-inspector-bayes.git](https://github.com/YOUR_USERNAME/interaction-inspector-bayes.git)
    cd interaction-inspector-bayes
    ```
2.  **Save the Code:** Save the provided script as `interaction_inspector.py`.
3.  **Run the Script:**
    ```bash
    python interaction_inspector.py
    ```

### 3. Gameplay Instructions
1.  **Review Historic Cohort Trends:** Observe the base quantities of Safe vs. High-Risk interactions within your clinical training archive.
2.  **Process incoming Telemetry:** Follow the script as it tracks a high-stakes clinical combination: an *Anticoagulant* prescribed to a known *Slow Metabolizer*.
3.  **Verify Likelihood Outputs:** Track how individual category weights chain together to alter the final system prediction.
4.  **Confirm the Decision Support Intercept:** Verify if your probabilistic engine successfully blocked the order or permitted a toxic clinical oversight.

---

## 🧠 Code Structure Highlights

### Conditional Likelihood Extraction
The system aggregates category occurrences within specific class slices to determine separate standalone conditional probabilities.

```python
# Counting specific historical matches within the 'High-Risk' subset
risk_class_count = sum(1 for x in history if x["label"] == 1 and x["features"][0] == test_class)
risk_meta_count = sum(1 for x in history if x["label"] == 1 and x["features"][1] == test_metabolism)

p_class_given_risk = risk_class_count / risk_count if risk_count > 0 else 0
p_meta_given_risk = risk_meta_count / risk_count if risk_count > 0 else 0

