# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: MIT
#
# WARS-CI-DFA: Tribute Pipeline Compliance Tester
# ===============================================

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scikit_runux import RunuxClassifier

def test_pipeline():
    print("=========================================================")
    print("    scikit-runux Tribute Pipeline Integration Tester   ")
    print("=========================================================\n")
    print("  [+] Dynamic Package Route: PUBLIC (Gated Stub)")
    
    # Initialize Classifier
    clf = RunuxClassifier(
        hidden_layer_sizes=(64, 32),
        learning_rate=0.01,
        max_iter=10,
        batch_size=16
    )
    
    # Build standard scikit-learn pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    # Generate simple test dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 3, size=100)
    
    print("  [+] Verifying Gated Stub Exception Behavior...")
    try:
        pipeline.fit(X, y)
        print("  ❌ STUB INTEGRATION FAILED: Expected fit to raise NotImplementedError, but it succeeded!")
    except NotImplementedError as e:
        print(f"      -> Caught Expected Exception: {str(e)}")
        print("\n  🎉 GATED STUB INTEGRATION PASSED 100% compliance test!")
    except Exception as e:
        print(f"  ❌ STUB INTEGRATION FAILED: Unexpected exception raised: {str(e)}")
        raise e

if __name__ == "__main__":
    test_pipeline()
