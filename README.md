# scikit-runux: Pour l'Honneur de l'Esprit Humain 🇫🇷

> [!NOTE]
> **A Lifelong Scientific Testimonial & Open Source Tribute to Professor Olivier Grisel**
>
> Inspired by the lectures of **Olivier Grisel** at *l'École Polytechnique* (X), the mathematical legacy of **Jean Dieudonné**, and the historic French engineering style (*l'art de l'ingénieur français*).

---

## 1. The Philosophical & Mathematical Genesis

This repository is an open-source tribute hosting the public interfaces and formal specifications of `scikit-runux`—a biomimetic, backpropagation-free machine learning extension for the *Scikit-Learn* ecosystem.

### The French Engineering Style & Mathematical Passion
In the French tradition of mathematical engineering, software code is not merely a utility; it is a formal canvas for mathematical beauty, absolute logical rigor, and aesthetic structural elegance. This philosophy unites:
*   **Deep Mathematical Abstraction**: Grounded in the Bourbakist tradition of building clean, first-principles architectures.
*   **Aesthetic & Frugal Design**: Constructing highly optimized codebases that leverage physical vector units and cache boundaries.
*   **Formal Integrity**: Guaranteeing convergence using proof assistants like *Lean 4*.

### For the Honor of the Human Spirit
At the age of twelve, the author was deeply influenced by Jean Dieudonné’s seminal book, ***Pour l'honneur de l'esprit humain*** (*For the Honor of the Human Spirit*). Dieudonné argued that the search for mathematical truth is one of the highest honors of human consciousness. 

Years later, during studies at *l'École Polytechnique*, this belief met the institution's historical motto:

$$\text{\bf Pour la Patrie, les Sciences et la Gloire}$$
$$\text{(For the Fatherland, Science, and Glory)}$$

This tribute is born from that alignment: a lifelong scientific dedication to designing technology that honors human intelligence through the rigorous pursuit of science.

---

## 2. Dedication to Olivier Grisel

We dedicate this framework to **Professor Olivier Grisel**, an alumnus of *l'École Polytechnique* and a key *Scikit-Learn* core contributor at *INRIA*. 

Through his exceptional lectures, profound design insights, and tireless work maintaining the open-source machine learning ecosystem, Olivier has inspired generations of engineers. His career embodies the absolute best of the French engineering tradition—combining high-level mathematical theory with high-impact, democratized open-source code.

---

## 3. Package Integration

`scikit-runux` integrates seamlessly into standard Scikit-Learn pipelines.

### Installation
```bash
pip install -e .
```

### Standard Estimator Usage
```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scikit_runux import RunuxClassifier

# Standard, pipeline-compliant biomimetic classification
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RunuxClassifier(
        hidden_layer_sizes=(128, 64),
        learning_rate=0.005,
        max_iter=40
    ))
])
```

---

## 4. WARS-CI-DFA Architectural Principles

`scikit-runux` bypasses standard Backpropagation completely, utilizing **WARS Co-Inference Direct Feedback Alignment (CI-DFA)**:
1.  **Direct Feedback Projections**: Updates are executed locally during the forward pass by projecting global errors through fixed random matrices $B_i$.
2.  **Telemetry-Gated Synaptic Pruning (TG-SP)**: Modulates weight updates based on real-time cache telemetry metrics, pruning non-essential synaptic updates under compute pressure.
3.  **Formal Convergence Proofs**: Proved and closed in the *Lean 4* mathematical specification under Section 6 of `RunuX.lean`.

---

## License

This tribute repository is open-sourced under the [MIT License](LICENSE) to support open scientific inquiry. The full, optimized high-performance training kernels are proprietary under **Socrate AI Lab** (Patent Pending: `US-PAT-PEND-2026-0525`).
