# Copyright (c) 2026 Xavier Callens / Socrate AI Lab. All Rights Reserved.
# SPDX-License-Identifier: MIT
#
# Biomimetic Co-Inference Training: Logic Tensor Network Fuzzy Constraints
# ======================================================================

import numpy as np
from typing import List

class BiomimeticFuzzyLogicGatekeeper:
    """
    Implements Logic Tensor Network (LTN) fuzzy logic predicates
    to audit biologically-inspired training safety boundaries.
    """
    def __init__(self, beta: float = 5.0):
        self.beta = beta

    def weights_bounded(self, W_list: List[np.ndarray], limit: float = 100.0) -> float:
        """
        Fuzzy predicate: weights_bounded(W)
        Evaluates the truth value in [0.0, 1.0] that the Frobenius norm of all weight matrices is bounded.
        $$I(\\text{weights\\_bounded}(W)) = \\prod_i e^{-\\beta \\max(0, \\|W_i\\|_F - \\text{limit})}$$
        """
        satisfaction = 1.0
        for W in W_list:
            norm = np.linalg.norm(W)
            excess = max(0.0, norm - limit)
            truth = np.exp(-self.beta * excess)
            satisfaction *= truth
        return float(np.clip(satisfaction, 0.0, 1.0))

    def error_converging(self, loss_history: List[float], convergence_threshold: float = 0.5) -> float:
        """
        Fuzzy predicate: error_converging(e)
        Evaluates the truth value in [0.0, 1.0] that the training loss converges towards zero.
        $$I(\\text{error\\_converging}(e)) = e^{-\\beta \\max(0, L_{\\text{final}} - L_{\\text{initial}} \\cdot \\text{threshold})}$$
        """
        if not loss_history or len(loss_history) < 2:
            return 0.0
            
        initial_loss = loss_history[0]
        final_loss = loss_history[-1]
        
        # Loss must decrease below a fraction of initial loss
        target_max_loss = initial_loss * convergence_threshold
        excess_loss = max(0.0, final_loss - target_max_loss)
        
        # Also check for NaNs/Infs
        if np.isnan(final_loss) or np.isinf(final_loss):
            return 0.0
            
        truth = np.exp(-self.beta * excess_loss)
        return float(np.clip(truth, 0.0, 1.0))

    def energy_drift_satisfaction(self, energy_history: List[float], drift_threshold: float = 1e-10) -> float:
        """
        Fuzzy predicate: energy_conservation(E)
        Evaluates the truth value in [0.0, 1.0] that the physical energy is conserved.
        $$I(\\text{energy\\_conservation}(E)) = e^{-\\beta \\max(0, \\text{max\\_drift} - \\text{drift\\_threshold})}$$
        """
        if not energy_history:
            return 0.0
            
        E0 = energy_history[0]
        if abs(E0) < 1e-30:
            return 1.0
            
        max_drift = 0.0
        for E in energy_history:
            drift = abs(E - E0) / abs(E0)
            if drift > max_drift:
                max_drift = drift
                
        excess_drift = max(0.0, max_drift - drift_threshold)
        
        if np.isnan(max_drift) or np.isinf(max_drift):
            return 0.0
            
        truth = np.exp(-self.beta * excess_drift)
        return float(np.clip(truth, 0.0, 1.0))

    def helicity_drift_satisfaction(self, helicity_history: List[float], drift_threshold: float = 0.05) -> float:
        """
        Fuzzy predicate: helicity_preservation(H)
        Evaluates the truth value in [0.0, 1.0] that magnetic helicity is preserved.
        $$I(\\text{helicity\\_preservation}(H)) = e^{-\\beta \\max(0, \\text{max\\_drift} - \\text{drift\\_threshold})}$$
        """
        if not helicity_history:
            return 0.0
            
        H0 = helicity_history[0]
        max_drift = 0.0
        for H in helicity_history:
            # Normalize by 1e-5 to prevent near-zero relative drift explosion
            drift = abs(H - H0) / max(abs(H0), 1e-5)
            if drift > max_drift:
                max_drift = drift
                
        excess_drift = max(0.0, max_drift - drift_threshold)
        
        if np.isnan(max_drift) or np.isinf(max_drift):
            return 0.0
            
        truth = np.exp(-self.beta * excess_drift)
        return float(np.clip(truth, 0.0, 1.0))

    def evaluate_global_satisfaction(self, *predicates: float) -> float:
        """
        Evaluates the global fuzzy logic satisfaction of a conjunction of predicates using the Product t-norm:
        $$I(\\phi_1 \\land \\phi_2 \\land \\dots \\land \\phi_n) = \\prod_i I(\\phi_i)$$
        """
        satisfaction = 1.0
        for p in predicates:
            satisfaction *= p
        return float(satisfaction)
