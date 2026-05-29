def naive_bayes_pharmacy_game():
    # 1. Scenario: Pharmacy Clinical Decision Support System
    print("--- 🩺 THE INTERACTION INSPECTOR: NAIVE BAYES ANALYZER 🩺 ---")
    print("Mission: Scan incoming drug profiles for high-risk metabolic interactions.")
    print("Goal: Use historical category probabilities to evaluate a new prescription.")

    # 2. Historical Training Records (Categorical Matrix)
    # Features: [Drug Class, Metabolic Profile] -> Label: 0 = Safe, 1 = High Risk
    history = [
        {"features": ["NSAID", "Fast"], "label": 0},
        {"features": ["NSAID", "Slow"], "label": 0},
        {"features": ["Anticoagulant", "Fast"], "label": 0},
        {"features": ["Anticoagulant", "Slow"], "label": 1},
        {"features": ["Anticoagulant", "Slow"], "label": 1},
    ]
    
    total_records = len(history)
    safe_count = sum(1 for x in history if x["label"] == 0)
    risk_count = sum(1 for x in history if x["label"] == 1)

    print("\n--- 🖥️ HISTORICAL COHORT BASELINES ---")
    print(f"Total Reviewed Cases: {total_records}")
    print(f" -> Safe Alignments: {safe_count}")
    print(f" -> High-Risk Incidents: {risk_count}")

    # 3. Game Inputs: Simulating an Unknown Incoming Prescription Order
    # Let's say a doctor prescribes an "Anticoagulant" to a "Slow" metabolizer patient.
    test_class = "Anticoagulant"
    test_metabolism = "Slow"

    print(f"\n--- 🚨 DISPENSING QUEUE: NEW PRESCRIPTION ORDER RECEIVED ---")
    print(f"Checking compatibility for: Class = {test_class} | Patient Metabolism = {test_metabolism}")

    # 4. The Math: Computing Naive Bayes Probabilities
    print("\n--- 🔄 COMPUTING PROBABILITY SCALARS (BAYES MATRIX) ---")

    # Step A: Prior Probabilities P(Class)
    prior_safe = safe_count / total_records
    prior_risk = risk_count / total_records
    print(f"Prior Probability P(Safe): {prior_safe:.2f}")
    print(f"Prior Probability P(High-Risk): {prior_risk:.2f}")

    # Step B: Conditional Probabilities P(Feature | Class)
    # Count occurrences within the 'Safe' subset
    safe_class_count = sum(1 for x in history if x["label"] == 0 and x["features"][0] == test_class)
    safe_meta_count = sum(1 for x in history if x["label"] == 0 and x["features"][1] == test_metabolism)
    
    # Count occurrences within the 'High-Risk' subset
    risk_class_count = sum(1 for x in history if x["label"] == 1 and x["features"][0] == test_class)
    risk_meta_count = sum(1 for x in history if x["label"] == 1 and x["features"][1] == test_metabolism)

    # Calculate likelihoods (with a basic check to prevent division by zero)
    p_class_given_safe = safe_class_count / safe_count if safe_count > 0 else 0
    p_meta_given_safe = safe_meta_count / safe_count if safe_count > 0 else 0

    p_class_given_risk = risk_class_count / risk_count if risk_count > 0 else 0
    p_meta_given_risk = risk_meta_count / risk_count if risk_count > 0 else 0

    print(f"Likelihood P({test_class} | Safe): {p_class_given_safe:.2f}")
    print(f"Likelihood P({test_metabolism} | Safe): {p_meta_given_safe:.2f}")
    print(f"Likelihood P({test_class} | High-Risk): {p_class_given_risk:.2f}")
    print(f"Likelihood P({test_metabolism} | High-Risk): {p_meta_given_risk:.2f}")

    # Step C: Naive Assumption Combination
    # Multiply prior by all independent feature conditional likelihoods
    score_safe = prior_safe * p_class_given_safe * p_meta_given_safe
    score_risk = prior_risk * p_class_given_risk * p_meta_given_risk

    # Normalize values into relative percentages
    total_score = score_safe + score_risk
    prob_safe = score_safe / total_score if total_score > 0 else 0
    prob_risk = score_risk / total_score if total_score > 0 else 0

    print(f"\n--- 📊 CALCULATED POSTERIOR PROBABILITIES ---")
    print(f"Posterior Probability of being SAFE: {prob_safe:.2%}")
    print(f"Posterior Probability of HIGH-RISK INTERACTION: {prob_risk:.2%}")

    # 5. Output Verdict
    if prob_risk > prob_safe:
        prediction = 1
        verdict = "❌ ORDER REJECTED: HIGH-RISK DRUG INTERACTION DETECTED"
    else:
        prediction = 0
        verdict = "✅ ORDER APPROVED: DOSAGE COMBINATION DEEMED COMPATIBLE"

    print(f"\nInformatics Action: {verdict}")

    # 6. Validation Verification
    actual_truth = 1
    if prediction == actual_truth:
        print("\n🏆 SUCCESS: Your Naive Bayes filter prevented a hazardous patient drug conflict!")
        print("The clinical staff has been alerted to modify the therapeutic plan.")
    else:
        print("\n💥 CLINICAL ALARM: False classification calculated. Review your matrix assumptions!")

if __name__ == "__main__":
    naive_bayes_pharmacy_game()
