# Part 3 VIVA Q&A: Deep Tabular Learning, Ensembling & Explainability

This document provides comprehensive answers to anticipated questions for the Part 3 Viva, covering architecture, mathematics, evaluation, and interpretability.

---

### Q1: What is TabNet, and why use it instead of standard Multi-Layer Perceptrons (MLPs) for tabular financial data?
**Answer:**
Standard MLPs use fully connected dense layers, where every input feature interacts with every neuron. Tabular datasets, however, are characterized by:
1. Sparse feature relationships (only a few features matter for any given prediction).
2. Heterogeneous feature distributions (returns, ratios, volatilities).
3. Susceptibility to overfitting when fully connected networks memorize spurious correlations.

TabNet (Google Research, 2021) overcomes this by introducing:
- **Sequential Multi-Step Architecture**: Processes data in sequential decision steps ($N_{\text{steps}}=3$), mimicking decision trees.
- **Sparse Feature Selection**: Uses Sparsemax to select sparse subsets of features at each step, zeroing out irrelevant noise.
- **Prior Scale**: Tracks past feature usage, forcing subsequent steps to discover complementary signals.
In our experiments, TabNet was the only individual model to achieve a positive out-of-sample $R^2$ (+0.0108), outperforming XGBoost.

---

### Q2: How does Sparsemax differ from Softmax, and why is it critical for TabNet?
**Answer:**
Softmax computes $\frac{e^{z_i}}{\sum e^{z_j}}$, which asymptotically approaches zero but never reaches exact zero. Every feature receives a non-zero weight, creating dense gradients and muddying feature attribution.

Sparsemax (Martins & Astudillo, 2016) projects the logits directly onto the probability simplex via Euclidean projection:
$$\text{Sparsemax}(\mathbf{z}) = \arg\min_{\mathbf{p} \in \Delta} \|\mathbf{p} - \mathbf{z}\|^2$$
This projection clips small activations to **exact zeros**. Consequently, at each decision step, TabNet selects a truly sparse subset of features, exactly replicating the sparse splits of tree algorithms while remaining fully differentiable.

---

### Q3: What is Ghost Batch Normalization, and why is it used in TabNet?
**Answer:**
Ghost Batch Normalization (GBN) splits a large training batch (e.g., 256) into smaller "virtual" micro-batches (e.g., 64) and computes batch normalization statistics independently on each micro-batch.
- **Benefits in Tabular Data**: It introduces stochastic regularization into normalization statistics, preventing deep networks from overfitting to specific clusters of tabular samples.
- **In our configuration**: `batch_size=256` with `virtual_batch_size=64`.

---

### Q4: Why did we standardize the regression target for TabNet?
**Answer:**
Daily index returns are minute values with standard deviations around $\sim 0.0086$ (86 basis points). When training neural networks with Mean Squared Error (MSE), the gradients are proportional to $(y - \hat{y})$, producing tiny gradient magnitudes ($\sim 10^{-4}$). This causes weights to stall, and predictions collapse to a constant near zero.
By standardizing:
$$\tilde{y} = \frac{y - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
the target has zero mean and unit variance, providing healthy gradient flow through the network. At prediction time, predictions are rescaled: $\hat{y} = \hat{\tilde{y}} \cdot \sigma_{\text{train}} + \mu_{\text{train}}$.

---

### Q5: How was early stopping implemented, and why was it monitored on the Validation set rather than the Test set?
**Answer:**
Early stopping monitors the loss on the chronological **Validation set** (Nov 2021 to Jan 2024) with a patience of 20 epochs. If validation loss does not improve for 20 consecutive epochs, training terminates and weights from the best epoch are restored.
- **Integrity requirement**: Monitoring early stopping on the test set is a serious form of **data leakage**, as it tunes the stopping threshold using future test data. The test set was touched exactly once during final evaluation.

---

### Q6: How does the Ensemble work, and why did it achieve a lower MAE than either model alone?
**Answer:**
We implemented an optimal convex combination:
$$\hat{y}_{\text{ens}} = w^* \hat{y}_{\text{TabNet}} + (1 - w^*) \hat{y}_{\text{XGBoost}}$$
where $w^*$ was optimized on the validation split. The optimal weight was $w^* = 0.62$ (62% TabNet, 38% XGBoost).
- **Mathematical justification**: The error correlation between TabNet and XGBoost residuals was $\rho = 0.584$. Because $\rho < 1.0$, the variance of their combination is strictly less than their individual error variances:
$$\sigma_{\text{ens}}^2 < w^2 \sigma_1^2 + (1-w)^2 \sigma_2^2 + 2w(1-w)\rho \sigma_1 \sigma_2$$
Because TabNet (manifold projection) and XGBoost (axis-aligned orthogonal splits) have distinct structural inductive biases, their errors partially cancel out, yielding an out-of-sample MAE of **0.006211**, lower than TabNet (0.006221) and XGBoost (0.006215).

---

### Q7: What are TabNet Attention Masks $M[i]$, and how do they provide intrinsic explainability?
**Answer:**
At each decision step $i$, the Attentive Transformer outputs a selection mask $\mathbf{M}[i] \in \mathbb{R}^{B \times D}$. The entries represent the proportion of attention allocated to each feature.
- Unlike post-hoc explainers (like LIME) which approximate model behavior locally, TabNet's masks represent the exact internal routing of signals through the network.
- In our model:
  - Step 0 focused on macro cycle (`month_cos`: 0.309) and fast momentum (`ema_12_ratio`: 0.415).
  - Step 1 focused on trend anchoring (`ema_26_ratio`: 0.460).
  - Step 2 focused on intra-week calendar adjustments (`day_of_week`: 0.505).

---

### Q8: What is SHAP, and how does TreeSHAP differ from KernelSHAP?
**Answer:**
SHAP (Lundberg & Lee, 2017) computes additive feature attributions based on cooperative game theory:
$$\hat{f}(x) = \phi_0 + \sum_{j=1}^M \phi_j(x)$$
where $\phi_j(x)$ satisfies four fundamental axioms: **Efficiency** (sum of attributions equals output minus expected value), **Symmetry**, **Dummy** (zero marginal contribution gets zero attribution), and **Additivity**.
- **TreeSHAP**: An exact polynomial-time algorithm ($O(T L D^2)$) specifically designed for tree ensembles. It recursively traverses tree paths to evaluate all feature coalitions in a single pass.
- **KernelSHAP**: A model-agnostic sampling method that formulates a weighted linear regression problem around a sample using a background dataset. It is computationally more demanding and was applied to TabNet using a $k$-means summary ($k=8$) of the training background.

---

### Q9: Why is out-of-sample directional accuracy for daily index returns typically around 50–52%?
**Answer:**
Under the semi-strong form of the **Efficient Market Hypothesis (EMH)**, liquid index prices (like the NIFTY 50) rapidly incorporate publicly available historical price and volume data. Consequently, next-day returns are predominantly white noise with a minute drift.
- A directional accuracy of 51.3% (achieved by TabNet) is commercially meaningful in quantitative trading: over 698 trading days, a 1.3% edge over random walk compounds into substantial Sharpe ratios when paired with proper volatility sizing and risk management.

---

### Q10: Why did we exclude raw price levels (`open`, `high`, `low`, `close`) from the feature set?
**Answer:**
Raw price levels are non-stationary:
- During the training period (2007–2021), NIFTY 50 traded between ~2,500 and 15,315.
- During the test period (2024–2026), NIFTY 50 traded above 21,000 to >24,000.
Decision trees split on orthogonal thresholds and cannot extrapolate beyond their highest observed training value ($15,315$). Neural networks receive input activations multiple standard deviations outside their training normalization bounds, destabilizing activations.
Using stationary transformations (returns, SMA/EMA ratios, MACD ratio, rolling volatility, RSI) ensures features have stable distributions across all time periods.
