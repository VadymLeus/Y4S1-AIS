import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('renfe_small.csv')

df_clean = df.dropna(subset=['price', 'train_type', 'fare']).copy()

df_clean['train_type_code'] = pd.Categorical(df_clean['train_type']).codes
df_clean['fare_code'] = pd.Categorical(df_clean['fare']).codes

train_type_labels = pd.Categorical(df_clean['train_type']).categories
fare_labels = pd.Categorical(df_clean['fare']).categories

print("Дані успішно підготовлені.")
print(f"Категорії типів поїздів: {list(train_type_labels)}")
print(f"Категорії тарифів: {list(fare_labels)}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(df_clean['price'], kde=True)
plt.title('Розподіл цін на квитки')
plt.xlabel('Ціна (€)')
plt.ylabel('Частота')

plt.subplot(1, 2, 2)
sns.boxplot(x='train_type', y='price', data=df_clean)
plt.title('Ціна залежно від типу поїзда')
plt.xlabel('Тип поїзда')
plt.ylabel('Ціна (€)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

with pm.Model() as renfe_model:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta_train_type = pm.Normal('beta_train_type', mu=0, sigma=5, shape=len(train_type_labels))
    beta_fare = pm.Normal('beta_fare', mu=0, sigma=5, shape=len(fare_labels))
    alpha = pm.Exponential('alpha', 1.0)

    mu = pm.math.exp(beta0 + beta_train_type[df_clean['train_type_code']] + beta_fare[df_clean['fare_code']])
    
    beta = alpha / mu
    
    price = pm.Gamma('price', alpha=alpha, beta=beta, observed=df_clean['price'])

with renfe_model:
    idata = pm.sample(2000, tune=1000, cores=1)
    
    idata.extend(pm.sample_posterior_predictive(idata))

print("Модель успішно навчена.")

summary = az.summary(idata, var_names=['beta0', 'beta_train_type', 'beta_fare', 'alpha'], hdi_prob=0.94)
print("\nЗведення параметрів моделі:")
print(summary)

summary_train_type = az.summary(idata, var_names=['beta_train_type'], hdi_prob=0.94)
summary_train_type.index = [f"beta_train_type[{label}]" for label in train_type_labels]
print("\nЗведення коефіцієнтів типу поїзда:")
print(summary_train_type)

summary_fare = az.summary(idata, var_names=['beta_fare'], hdi_prob=0.94)
summary_fare.index = [f"beta_fare[{label}]" for label in fare_labels]
print("\nЗведення коефіцієнтів тарифу:")
print(summary_fare)

mean_coeffs_train = idata.posterior['beta_train_type'].mean(dim=('chain', 'draw')).values
print(f"\nБазовий тип поїзда: {train_type_labels[0]}")
for i, label in enumerate(train_type_labels[1:], 1):
    effect = np.exp(mean_coeffs_train[i] - mean_coeffs_train[0])
    print(f"Поїзд типу '{label}' у {effect:.2f} разів дорожчий/дешевший, ніж '{train_type_labels[0]}'")

az.plot_ppc(idata, num_pp_samples=100)
plt.title('Апостеріорна предиктивна перевірка')
plt.xlabel('Ціна (€)')
plt.show()

az.plot_forest(idata, var_names=['beta_train_type', 'beta_fare'], combined=True, hdi_prob=0.94)
plt.title('Апостеріорні розподіли для коефіцієнтів')
plt.show()





