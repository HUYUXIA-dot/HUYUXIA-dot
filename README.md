age = data['Age']
nep = data['NEP']

# def model(age, e, a, b):
#     return e + a * (1 - np.exp(b * age))

def model(age, a, b, c, d):
    return a * (1 + ((b * (age/c)**d - 1) / np.exp(age/c)))

# initial_guess = [-400, 700, -0.1]
initial_guess = [175, 12, 10.034, 1.005]


params, covariance = curve_fit(model, age, nep, p0=initial_guess)

# e, a, b = params
# print(f"Estimated parameters: e = {e}, a = {a}, b = {b}")
a, b, c, d = params
print(f"Estimated parameters: a = {a}, b = {b}, c = {c}, d = {d}")

# Create a range of ages for plotting the fitted curve
age_range = np.linspace(min(age), max(age), 100)

# Calculate the fitted NEP values
fitted_nep = model(age_range, *params)
fitted_nep1 = model(age, *params)

if len(nep) == len(fitted_nep1):
    # 计算 R^2
    slope, intercept, r_value, p_value, std_err = linregress(nep, fitted_nep1)
    r_squared = r_value**2

    # 计算 MAE (不使用外部库)
    mae = np.mean(np.abs(nep - fitted_nep1))

    # 计算 AIC
    residuals = nep - fitted_nep1
    sse = np.sum(residuals**2)
    aic = len(age) * np.log(sse / len(age)) + 2 * len(params)

    print(f"R^2: {r_squared}")
    print(f"MAE: {mae}")
    print(f"AIC: {aic}")
else:
    print("Error: Mismatched array lengths.")

# 设置全局字体为'New Times Roman'
plt.rcParams['font.family'] = 'Times New Roman'
# Plotting the data and the fitted curve
plt.figure(figsize=(10, 6))
plt.scatter(age, nep, color='orange')
plt.plot(age_range, fitted_nep, color='#5B9BD5', linewidth=2)
# plt.xlabel('森林年龄(年)')
# plt.ylabel('NEP（gC m$^{-2}$）')
# plt.title('Forest Age vs AGB with Fitted Curve')
plt.legend()
plt.show()

from scipy.stats import t


std_errs = np.sqrt(np.diag(covariance))

num_params = len(params)


t_values = params / std_errs

df = len(age) - num_params


p_values = [2 * (1 - t.cdf(np.abs(t_val), df)) for t_val in t_values]

# for param, t_val, p_val in zip(['e', 'a', 'b'], t_values, p_values):
#     print(f"参数 {param}: t 值 = {t_val}, p 值 = {p_val}")
for param, t_val, p_val in zip(['a', 'b', 'c', 'd'], t_values, p_values):
    print(f"参数 {param}: t 值 = {t_val}, p 值 = {p_val}")
