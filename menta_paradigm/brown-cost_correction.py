import numpy as np
import scipy.stats as stats

# P-values
p_values = [0.015, 0.002, 0.002, 0.016]
k = len(p_values)

# Calculate chi-square statistic
chi2_stat = -2 * sum(np.log(p_values))

# For df_eff = 2
df_eff_1 = 4
c_1 = 2 * k / df_eff_1
adjusted_chi2_1 = chi2_stat / c_1
brown_p_1 = stats.chi2.sf(adjusted_chi2_1, df_eff_1)

# For df_eff = 2.5
df_eff_2 = 2.5
c_2 = 2 * k / df_eff_2
adjusted_chi2_2 = chi2_stat / c_2
brown_p_2 = stats.chi2.sf(adjusted_chi2_2, df_eff_2)

print(f"P-values: {p_values}")
print(f"Chi-square statistic: {chi2_stat}")
print("\nWith df_eff = 2:")
print(f"Scaling factor (c): {c_1}")
print(f"Adjusted chi-square: {adjusted_chi2_1}")
print(f"Brown-adjusted p-value: {brown_p_1}")
print("\nWith df_eff = 2.5:")
print(f"Scaling factor (c): {c_2}")
print(f"Adjusted chi-square: {adjusted_chi2_2}")
print(f"Brown-adjusted p-value: {brown_p_2}")