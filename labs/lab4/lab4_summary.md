# Lab 4 Summary
## Key Statistics (Concrete)
 count  mean  median  mode   std  variance  min   q1  q2_median     q3  max  range   iqr  skewness  kurtosis
   100 35.06    34.2  31.8 6.163    37.979 21.8 31.1       34.2 39.125 47.1   25.3 8.025     0.047    -0.675

## Key Statistics by Material Type
               count     mean     std    min  q2_median    max  variance
material_type                                                           
Aluminum          50  196.632   7.633  186.2     194.50  212.4    58.267
Composite         50  299.042  10.384  281.5     298.70  319.2   107.837
Concrete          50   36.258   2.219   32.3      36.20   40.1     4.922
Steel             50  399.098  10.398  381.6     399.05  419.2   108.127

## Probability Results
- **Binomial P(X=3) [n=100,p=0.05]**: 0.139576
- **Binomial P(X<=5) [n=100,p=0.05]**: 0.615999
- **Poisson P(X=8) [λ=10]**: 0.112599
- **Poisson P(X>15) [λ=10]**: 0.048740
- **Normal P(X>280) [μ=250,σ=15]**: 0.022750
- **Normal 95th percentile [μ=250,σ=15]**: 274.672804
- **Exponential P(T<=500) [mean=1000]**: 0.393469
- **Exponential P(T>1500) [mean=1000]**: 0.223130
- **Bayes PPV P(Damage|+)**: 0.333333
- **Bayes NPV P(¬Damage|-)**: 0.997085

## Generated Plots
- **Concrete Strength Distribution** → `concrete_strength_distribution.png`
- **Concrete Strength Boxplot** → `concrete_strength_boxplot.png`
- **Distribution Fitting** → `distribution_fitting.png`
- **Probability Distributions** → `probability_distributions.png`
- **Bayes Probability Tree** → `bayes_probability_tree.png`
- **Material Comparison Boxplot** → `material_comparison_boxplot.png`
- **Statistical Summary Dashboard** → `statistical_summary_dashboard.png`

## Engineering Interpretations
- Higher variability (std, IQR) suggests conservative design choices.
- Percentiles help set acceptance criteria (e.g., 5th/95th for strengths/loads).
- PPV/NPV guide decisions after diagnostic tests (repair vs. monitor).