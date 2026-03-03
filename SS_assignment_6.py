import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Dataset
data = {
    'ctr': [3.2, 5.8, 2.1, 7.4, 4.5, 6.2, 1.8, 5.1, 3.9, 8.5,
            4.2, 2.8, 6.8, 3.5, 7.1, 2.4, 5.5, 4.8, 6.5, 3.1,
            7.8, 2.6, 5.3, 4.1, 6.9, 3.7, 8.1, 2.2, 5.9, 4.6],

    'total_views': [12000, 28000, 8500, 42000, 19000, 33000, 7000, 24000, 16000, 51000,
                    18000, 11000, 38000, 14500, 44000, 9500, 26000, 21000, 35000, 13000,
                    47000, 10000, 25000, 17500, 40000, 15000, 49000, 8000, 31000, 20000]
}

df = pd.DataFrame(data)

print("first 5 rows:\n")
print(df.head())
print("\n basic statistics: \n")
print(df.describe())

x=df[['ctr']]
y=df['total_views']

x_train,x_test,y_train,y_test=train_test_split(x, y, random_state=44, test_size=0.2)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print("\ncheck parameters\n")
print(f"coefficient: {model.coef_[0]:.2f}")
print(f"intercept: {model.intercept_:.2f}")

print("actual vs predictions")
print(pd.DataFrame({'Actual':y_test.values, 'Predictions':y_pred.round()}))

print(f"Arjun's question: If my thumbnail gets a 8% CTR, how many views can I expect?")
new_ctr= 8
new_views=model.predict([[8]])
print(f"{new_views[0]:.2f}")

plt.figure(figsize=(10,6))
plt.scatter(df['ctr'], df['total_views'], c=df['total_views'], cmap='viridis', s=100, alpha=0.7, edgecolors='white')
xline=np.linspace(df['ctr'].min(), df['ctr'].max(),100).reshape(-1,1)
plt.plot(xline, model.predict(xline), color='red', linewidth=2.5)
plt.scatter(new_ctr, new_views, color='green', s=125, alpha=0.7)
plt.xlabel('ctr', fontsize=12, fontweight='bold')
plt.ylabel('total_views', fontsize=12, fontweight='bold')
plt.title('ctr vs total views',fontsize=12, fontweight='bold')
plt.tight_layout()
plt.colorbar(label='views intensity')
plt.grid(True, alpha=0.3)
plt.show()



# Dataset
data = {
    'distance_km': [2.5, 6.0, 1.2, 8.5, 3.8, 5.2, 1.8, 7.0, 4.5, 9.2,
                    2.0, 6.5, 3.2, 7.8, 4.0, 5.8, 1.5, 8.0, 3.5, 6.8,
                    2.2, 5.5, 4.2, 9.0, 2.8, 7.2, 3.0, 6.2, 4.8, 8.2],

    'prep_time_min': [10, 20, 8, 25, 12, 18, 7, 22, 15, 28,
                      9, 19, 11, 24, 14, 17, 6, 26, 13, 21,
                      10, 16, 14, 27, 11, 23, 12, 18, 15, 25],

    'delivery_time_min': [18, 38, 12, 52, 24, 34, 14, 45, 29, 58,
                          15, 40, 21, 50, 27, 35, 11, 54, 23, 43,
                          17, 32, 26, 56, 19, 47, 20, 37, 30, 53]
}

df = pd.DataFrame(data)

print('first 5 rows:\n')
print(df.head())
print("\n Basic Statistics:\n")
print(df.describe())

names=['distance_km', 'prep_time_min']
fig,axes=plt.subplots(1, 2, figsize=(14,6), constrained_layout=True)
for i,name in enumerate(names):
    s=axes[i].scatter(df[name], df['delivery_time_min'], c=df['delivery_time_min'],cmap='plasma', s=100, alpha=0.7, edgecolors='white')
    axes[i].set_xlabel(name, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('delivery_time_min', fontsize=12, fontweight='bold')
    axes[i].set_title(f"{name} vs delivery_time_min", fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
plt.suptitle('ACtual data', fontsize=12, fontweight='bold')
plt.colorbar(s, ax=axes, label='delivery_time_intensity', pad=0.02)
plt.show()

x=df[['distance_km', 'prep_time_min']]
y=df['delivery_time_min']

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=44)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("check parameters:\n")
print(f"coefficients: distance_km:{model.coef_[0]:.2f}\n prep_time_min:{model.coef_[1]:.2f}")
print(f"intercept:{model.intercept_:.2f}")

print("ACtual vs predictions data:\n")
print(pd.DataFrame({'Actual':y_test.values, 'Predictions':y_pred.round()}))

print(f"new question is: If the distance is 7 km and preparation takes 15 minutes, what should I tell the customer?")
new_time=model.predict([[7, 15]])
print(f'{new_time[0]:.2f}')

fixed_values={'distance_km':df['distance_km'].mean(), 'prep_time_min':df['prep_time_min'].mean()}

fig,axes=plt.subplots(1, 2, figsize=(14,6), constrained_layout=True)
for i,name in enumerate(names):
    sa=axes[i].scatter(df[name],df['delivery_time_min'], c=df['delivery_time_min'],cmap='viridis', s=100, alpha=0.7, edgecolors='white')
    xline=np.linspace(df[name].min(), df[name].max(), 100)
    x_line=pd.DataFrame({'distance_km':fixed_values['distance_km'], 'prep_time_min': fixed_values['prep_time_min']},index=range(100))
    x_line[name]=xline
    axes[i].plot(xline, model.predict(x_line), color='red', linewidth=2.5)
    axes[i].set_xlabel(name, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('delivery_time_min', fontsize=12, fontweight='bold')
    axes[i].set_title(f"{name} vs delivery_time_min", fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
plt.suptitle('REGRESSION LINES', fontsize=14, fontweight='bold')
plt.colorbar(sa, ax=axes, label='delivery_time_Intensity', pad=0.02)
plt.show()

# Dataset
data = {
    'ram_gb': [4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16,
               4, 8, 4, 16, 8, 8, 4, 16, 8, 16],

    'storage_gb': [256, 512, 128, 512, 256, 512, 256, 1024, 256, 512,
                   128, 512, 256, 1024, 256, 512, 128, 512, 256, 1024,
                   256, 512, 128, 512, 256, 512, 256, 1024, 256, 512],

    'processor_ghz': [2.1, 2.8, 1.8, 3.2, 2.4, 3.0, 2.0, 3.5, 2.6, 3.0,
                      1.6, 2.8, 2.2, 3.4, 2.5, 2.9, 1.9, 3.1, 2.3, 3.6,
                      2.0, 2.7, 1.7, 3.3, 2.4, 3.0, 2.1, 3.5, 2.6, 3.2],

    'price_inr': [28000, 45000, 22000, 72000, 38000, 52000, 26000, 95000, 42000, 68000,
                  20000, 48000, 29000, 88000, 40000, 50000, 23000, 70000, 36000, 98000,
                  25000, 46000, 21000, 75000, 39000, 53000, 27000, 92000, 43000, 73000]
}

df = pd.DataFrame(data)

print('first 5 rows:\n')
print(df.head())
print('\nBasic Statistics:\n')
print(df.describe())

names=['ram_gb', 'storage_gb', 'processor_ghz']

fig, axes = plt.subplots(1, 3, figsize=(14,6),  constrained_layout=True)
for i,name in enumerate(names):
    s=axes[i].scatter(df[name], df['price_inr'], c=df['price_inr'], cmap='viridis', s=50, alpha=0.7, edgecolors='black', label=f"{names} data")
    axes[i].set_xlabel(name, fontweight='bold', fontsize=12)
    axes[i].set_ylabel('price_inr', fontweight='bold', fontsize=12)
    axes[i].set_title(f"{name} vs price_inr", fontweight='bold', fontsize=12)
    axes[i].grid(True, alpha=0.3)
plt.suptitle("ACTUAL DATA")
plt.colorbar(s, ax=axes, label='price intensity',pad=0.02)
plt.show()

x=df[['ram_gb', 'storage_gb', 'processor_ghz']]
y=df['price_inr']

x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=44)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print('check parameters:\n')
print(f"coefficients= ram_gb:{model.coef_[0]:.2f}\nstorage_gb:{model.coef_[1]:.2f}\nprocessor_ghz:{model.coef_[1]:.2f}")
print(f"intercept: {model.intercept_:.2f}")

if model.coef_[0]>model.coef_[1] and model.coef_[0]>model.coef_[2]:
    print(f"ram_gb affects more in price")
elif model.coef_[1]>model.coef_[0] and model.coef_[1]>model.coef_[2]:
    print(f"storage_gb affects more in price")
elif model.coef_[2]>model.coef_[0] and model.coef_[2]>model.coef_[1]:
    print(f"processor_ghz affects more in price")


print(f'Actual vs Predictions')
print(pd.DataFrame({'Actual':y_test.values, 'Predictions':y_pred}))

score=r2_score(y_test,y_pred)
print(f'r2 score is : {score:.2f}')

print(f"new question is: for 16GB RAM, 512GB storage, 3.2 GHz processor, what is a fair price?")
new_price=model.predict([[16,512,3.2]])
print(f"{new_price[0]:.2f}")

print(f"Bonus Question is: a laptop with 8GB RAM, 512GB storage, 2.8 GHz for 55,000 INR. Is it overpriced?")
bonus_price=model.predict([[8,512,2.8]])
bonus_price=bonus_price[0]
if bonus_price>55000:
    print(f"it is overpriced")
elif bonus_price==55000:
    print(f"it is correct price")
else:
    print(f"it is below priced.")
print(f"difference is: {bonus_price-55000}")

fixed_values={'ram_gb':df['ram_gb'].mean(), 'storage_gb':df['storage_gb'].mean(), 'processor_ghz':df['processor_ghz'].mean()}
fig, axes = plt.subplots(1, 3, figsize=(14,6), constrained_layout=True)
for i,name in enumerate(names):
    sa=axes[i].scatter(df[name], df['price_inr'], c=df['price_inr'], cmap='viridis', s=100, alpha=0.7, edgecolors='black', label=f"{name} data")
    xline=np.linspace(df[name].min(), df[name].max(), 100).reshape(-1,1)
    x_line=pd.DataFrame({'ram_gb':fixed_values['ram_gb'], 'storage_gb': fixed_values['storage_gb'], 'processor_ghz':fixed_values['processor_ghz']},index=range(100))
    x_line[name]=xline
    axes[i].plot(xline,model.predict(x_line),color='red',linewidth=2.5)
    axes[i].set_xlabel(name, fontweight='bold', fontsize=12)
    axes[i].set_ylabel('price_inr', fontweight='bold', fontsize=12)
    axes[i].set_title(f"{name} vs price_inr", fontweight='bold', fontsize=12)
    axes[i].grid(True, alpha=0.3)
plt.suptitle("Regression Lines")
plt.colorbar(sa, ax=axes, label='price intensity',pad=0.02)
plt.show()
