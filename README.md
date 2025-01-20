# DS-tutorial-1

Residual Standard Error, R², and F-Statistics Analysis for Advertising Data

## Objective
Based on advertising data analyze the relationship between sales and advertising budgets for TV, Radio, and Newspaper. Using Ordinary Least Squares (OLS) regression, the following metrics are calculated:

1. **Residual Standard Error (RSE):** Measures the average deviation of observed values from predicted values.
2. **R² (R-squared):** Indicates the proportion of variance in sales explained by advertising budgets.
3. **F-statistics:** Tests the overall significance of the regression model.


## Results
The analysis produced the following outcomes:

TV
Residual Standard Error (RSE): 3.2423
R² : 0.6119
F-statistic: 1.4674e-42

Radio
Residual Standard Error (RSE): 4.2535
R² : 0.3320
F-statistic: 4.3550e-19

Newspaper
Residual Standard Error (RSE): 5.0670
R² : 0.0521
F-statistic: 0.0011

## Interpretation
1. **TV:**
   - The high R² (0.6119) indicates a strong correlation between TV advertising and sales.
   - The low RSE (3.2423) shows relatively accurate predictions.
   - The significant F-statistic (1.4674e-42) confirms the robustness of the model.

2. **Radio:**
   - The moderate R² (0.3320) implies a weaker relationship than TV.
   - The RSE (4.2535) indicates less precise predictions.
   - The F-statistic (4.3550e-19) confirms the model is still statistically significant.

3. **Newspaper:**
   - The very low R² (0.0521) suggests minimal influence of newspaper advertising on sales.
   - The high RSE (5.0670) reflects poor predictive accuracy.
   - The insignificant F-statistic (0.0011) suggests the model is not meaningful for this medium.

## Conclusion
- TV advertising is the most impactful medium for boosting sales.
- Radio advertising has a moderate effect.
- Newspaper advertising shows minimal to no significant impact on sales, based on this dataset.

