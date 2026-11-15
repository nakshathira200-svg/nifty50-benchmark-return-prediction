# Part 3 Methodology: Deep Learning (TabNet), Ensembling & Explainability (SHAP)

## 1. Executive Summary & Problem Scope

This document details the advanced modelling and explainability methodology for **Shanmuga Vel (PES2UG24AM083)** in the NIFTY 50 Benchmark Return Prediction project. Inheriting the cleaned, engineered, and chronologically partitioned datasets from Part 1 and traditional ML foundations from Part 2, Shanmuga Vel focuses on:
1. **PyTorch TabNet**: Implementing Google Research's canonical deep tabular architecture with sequential sparse attention.
2. **Head-to-Head Comparison**: Benchmarking TabNet against XGBoost (tuned gradient-boosted trees) and Decision Tree baselines on identical data and splits.
3. **Ensemble Learning**: Formulating and optimizing multi-paradigm ensembles (convex weighted blending and stacking) to test whether combining deep neural attention with tree-based boosting yields variance reduction.
4. **Comprehensive Interpretability**: Performing multi-level explainability through TabNet's native step-wise attention masks ($M[0], M[1], M[2]$) and Shapley additive explanations (TreeSHAP and KernelSHAP).

---

## 2. Chronological Splits & Anti-Leakage Protocol

Financial returns exhibit non-stationarity, regime shifts, and autocorrelation. Standard random $K$-fold cross-validation suffers from temporal look-ahead leakage. Therefore, this project enforces a strict **chronological partition**:

| Partition | Date Range | Trading Days | Proportion | Role in Pipeline |
|---|---|---|---|---|
| **Train** | Nov 2007 – Mar 2021 | 3,251 | ~70% | Fitting model weights and training standardizer |
| **Validation** | Mar 2021 – Jan 2024 | 695 | ~15% | Early stopping, hyperparameter grid selection, ensemble weight optimization |
| **Test** | Jan 2024 – Nov 2026 | 698 | ~15% | Strictly untouched, one-shot out-of-sample benchmark |

### Anti-Leakage Invariants Enforced in Code (`assert_no_leakage`):
1. **Chronological Ordering**: Each partition's index is strictly monotonically increasing.
2. **Temporal Separation**: $\max(\text{Train Date}) < \min(\text{Val Date})$ and $\max(\text{Val Date}) < \min(\text{Test Date})$.
3. **Boundary Row Purge**: Because the target is next-day return, the final row of Train (whose target extends into Validation) and Validation (whose target extends into Test) were purged in preprocessing.
4. **Train-Only Preprocessing**: The `StandardScaler` is fitted solely on the Train split and merely transformed across Validation and Test.
5. **Feature Set Restrictions**: The 4 raw price levels (`open`, `high`, `low`, `close`) are excluded by default in the `stationary` feature set (17 stationary features). Raw index prices grew from ~2,500 to >24,000, making unadjusted price levels severely out-of-distribution for neural networks and impossible for decision trees to extrapolate.
6. **Correlation Sanity Gate**: Every feature is verified to have $|r| < 0.5$ against targets on the training set, eliminating accidental target leakage.

---

## 3. PyTorch TabNet Architecture & Mathematical Formulation

TabNet (Arik & Pfister, *AAAI 2021*) is designed specifically for tabular data, combining the parameter efficiency of neural networks with the inductive bias of decision trees.

### 3.1 Sequential Multi-Step Processing
Rather than processing all features simultaneously with dense linear layers, TabNet processes tabular inputs in $N_{\text{steps}}$ sequential decision steps (we use $N_{\text{steps}} = 3$):
- **Decision Step 0**: Discovers broad regime indicators (e.g., trend ratios).
- **Decision Step 1**: Focuses on medium-term dynamics (moving average crossovers).
- **Decision Step 2**: Captures short-term volatility and calendar patterns.

### 3.2 The Attentive Transformer and Sparsemax
At each decision step $i$, the model dynamically selects which features to attend to using an **Attentive Transformer**:
$$\mathbf{M}[i] = \text{Sparsemax}\left(\mathbf{P}[i-1] \cdot \mathbf{h}_i(\mathbf{a}[i-1])\right)$$
where:
- $\mathbf{a}[i-1]$ is the processed feature representation from the preceding step.
- $\mathbf{h}_i$ is a trainable linear layer with Ghost Batch Normalization.
- $\mathbf{P}[i-1]$ is the **Prior Scale** matrix:
$$\mathbf{P}[i] = \prod_{j=1}^i (\gamma - \mathbf{M}[j])$$
The relaxation parameter $\gamma = 1.3$ penalizes features that have already been heavily utilized in earlier steps, enforcing feature exploration across steps.
- **Sparsemax** replaces standard Softmax:
$$\text{Sparsemax}(\mathbf{z}) = \arg\min_{\mathbf{p} \in \Delta} \|\mathbf{p} - \mathbf{z}\|^2$$
Unlike Softmax (which assigns non-zero probabilities to all inputs), Sparsemax projects directly onto the simplex $\Delta$, producing exact zeros. This induces true sparsity in feature selection, creating natural interpretability.

### 3.3 The Feature Transformer
The selected features $\mathbf{M}[i] \odot \mathbf{x}$ pass into the **Feature Transformer**, composed of:
1. Two **Shared Layers** (shared across all decision steps for common representation learning).
2. Two **Step-Dependent Layers** (unique to step $i$ for localized feature transformations).
Each layer consists of a fully connected block, Ghost Batch Normalization, and a **Gated Linear Unit (GLU)**:
$$\text{GLU}(\mathbf{x}) = (\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \otimes \sigma(\mathbf{x} \mathbf{W}_2 + \mathbf{b}_2)$$
GLU controls information flow and mitigates vanishing gradients.

### 3.4 Target Normalization for Financial Returns
Daily stock index returns are small quantities ($\mu \approx 0.0005, \sigma \approx 0.01$). Passing raw returns directly into neural network MSE loss functions causes gradients to vanish, often collapsing predictions to the training mean.
To solve this:
$$\tilde{y} = \frac{y - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Predictions $\hat{\tilde{y}}$ are un-standardized at inference:
$$\hat{y} = \hat{\tilde{y}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$$

---

## 4. Benchmark Models: XGBoost & Decision Tree

To provide rigorous context for TabNet, two reference models are trained:

### 4.1 XGBoost Reference Model
- **Algorithm**: Histogram-based Gradient Boosted Decision Trees (`tree_method="hist"`).
- **Hyperparameter Grid**: Tuned on the validation split:
  - `max_depth` $\in \{2, 3, 4\}$
  - `learning_rate` $\in \{0.02, 0.05\}$
- **Regularization**: $L_2$ leaf regularization $\lambda = 5.0$, `min_child_weight` = 5, `subsample` = 0.8, `colsample_bytree` = 0.8.
- **Early Stopping**: 50 rounds monitored on Validation RMSE (regression) and Validation Log-Loss (classification).

### 4.2 Traditional Decision Tree Baseline
- **Algorithm**: Single CART decision tree (as part of the Part 3 model-comparison experiments).
- **Hyperparameter Grid**: `max_depth` $\in \{2, 3, 4, 5, 6, 8\}$, tuned on the validation set.
- **Purpose**: Quantifies the marginal gain achieved by ensemble boosting (XGBoost) and deep attention (TabNet) over a traditional single-tree heuristic.

---

## 5. Ensemble Learning Formulation

A central research question in Shanmuga Vel is whether combining a tree-based ensemble (XGBoost) and a deep neural attention architecture (TabNet) yields performance superior to either model alone.

### 5.1 Mathematical Rationale: Error Decorrelation
Let $e_1 = y - \hat{y}_1$ and $e_2 = y - \hat{y}_2$ be the prediction errors of TabNet and XGBoost, with variances $\sigma_1^2, \sigma_2^2$ and correlation coefficient $\rho = \text{Corr}(e_1, e_2)$.
For an ensemble $\hat{y}_{\text{ens}} = w \hat{y}_1 + (1-w) \hat{y}_2$:
$$\sigma_{\text{ens}}^2 = w^2 \sigma_1^2 + (1-w)^2 \sigma_2^2 + 2w(1-w)\rho \sigma_1 \sigma_2$$
If $\rho < 1$, the variance of the ensemble is strictly lower than the convex combination of individual variances:
$$\sigma_{\text{ens}}^2 < w \sigma_1^2 + (1-w) \sigma_2^2$$
Because TabNet relies on continuous manifold embeddings and hyperplane projections while XGBoost relies on orthogonal axis-aligned partition trees, their error structures differ significantly ($\rho < 0.65$), creating a theoretical foundation for ensembling.

### 5.2 Implementation Strategies
1. **Weighted Average Blending**:
   $$\hat{y}_{\text{ens}} = w^* \hat{y}_{\text{TabNet}} + (1 - w^*) \hat{y}_{\text{XGBoost}}$$
   where $w^* \in [0, 1]$ is optimized via dense grid search on the **Validation** set to minimize Validation RMSE (for regression) or maximize Validation ROC-AUC (for classification).
2. **Stacking Meta-Learner**:
   A regularized Ridge regression model fit on validation predictions:
   $$\hat{y}_{\text{stack}} = \beta_0 + \beta_1 \hat{y}_{\text{TabNet}} + \beta_2 \hat{y}_{\text{XGBoost}}$$

---

## 6. Interpretability & Explainability Framework

Shanmuga Vel implements a dual explainability framework:

### 6.1 Intrinsic Explainability: TabNet Attention Masks
TabNet directly outputs the selection masks $\mathbf{M}[i] \in \mathbb{R}^{B \times D}$ for each decision step $i \in \{0, 1, 2\}$:
- **Aggregate Attention**:
  $$\mathbf{M}_{\text{agg}} = \sum_{i=0}^{N_{\text{steps}}-1} \eta_i \mathbf{M}[i]$$
  where $\eta_i$ is the aggregate decision weight of step $i$.
- This enables visualization of both global feature prioritization and localized sample-by-sample attention shifts.

### 6.2 Post-Hoc Explainability: SHAP (Shapley Additive exPlanations)
Based on cooperative game theory, the Shapley value $\phi_j$ allocates the marginal payout of feature $j$ across all possible feature coalitions $S \subseteq F \setminus \{j\}$:
$$\phi_j(x) = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{j\}) - f(S) \right]$$
- **TreeSHAP** (Lundberg et al., *Nature Machine Intelligence 2020*): Applied to XGBoost for exact polynomial-time ($O(T L D^2)$) Shapley value calculation.
- **KernelSHAP**: Applied to TabNet using a $k$-means summarized background distribution ($k=8$) to estimate Shapley values across out-of-sample test cases.
- **Visualizations Generated**:
  1. Beeswarm Summary Plot (feature value vs direction of impact).
  2. Relative Importance Bar Plot (mean $|\phi_j|$ comparison between TabNet and XGBoost).
  3. Dependence Plots (non-linear interaction curves for top predictive features).
