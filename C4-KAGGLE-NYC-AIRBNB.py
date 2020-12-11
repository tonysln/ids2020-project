# C4-KAGGLE-NYC-AIRBNB
# Authors: Anton Slavin, Elen Liivapuu


# Setup ===================================================
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
plt.style.use('ggplot')
# =========================================================


# === Initial data cleanup ================================
# Import data from the original file
print("Opening AB_NYC_2019.csv as a DataFrame...")
data = pd.read_csv("data/AB_NYC_2019.csv", index_col=0)
print("Initial size:",len(data), "x", len(data.columns))

# Remove the columns with a lot (20%) of missing values
data = data.drop(columns=['last_review', 'reviews_per_month'])

# Remove the rows with some missing values (37 rows)
data = data.dropna()

print("\nData:")
print(data)
# =========================================================


# === Visualizations ======================================
print("\nCreating visualizations for the dataset...")

plt.clf() # Reset plot
plt.title("Neighbourhood group relative frequency")
plt.xlabel("Neighbourhood group")
plt.ylabel("Frequency (%)")
(data['neighbourhood_group'].value_counts()/sum(data['neighbourhood_group'].value_counts())).plot(kind='bar')
plt.savefig('output/fig1.png', bbox_inches='tight')

plt.clf() # Reset plot
plt.title("Room type relative frequency")
plt.xlabel("Room type")
plt.ylabel("Frequency (%)")
(data['room_type'].value_counts()/sum(data['room_type'].value_counts())).plot(kind='bar')
plt.savefig('output/fig2.png', bbox_inches='tight')

plt.clf() # Reset plot
plt.title("Price distribution per room type")
sns.scatterplot(x ="room_type", y ="price", data = data)
plt.savefig('output/fig3.png', bbox_inches='tight')

plt.clf() # Reset plot
plt.title("Price distribution per neighbourhood group")
sns.scatterplot(x ="neighbourhood_group", y ="price", data = data)
plt.savefig('output/fig4.png', bbox_inches='tight')

# Display the mean price for listings based on the room type and neighbourhood group
plt.clf() # Reset plot
plt.title("Mean price for room type and neighbourhood group\n")
tab = pd.crosstab(data['room_type'],data['neighbourhood_group'], aggfunc='mean', values=data['price'])
sns.heatmap(tab,annot=True, fmt='g')
plt.savefig('output/fig5.png', bbox_inches='tight')

plt.clf() # Reset plot
plt.figure(figsize=(10,50))
plt.title("Mean price for neighbourhood")
plt.xlabel("Price (avg)")
plt.ylabel("Neighbourhood")
data['price'].groupby(data['neighbourhood']).mean().sort_values().plot(kind='barh')
plt.savefig('output/fig6.png', bbox_inches='tight')

plt.clf() # Reset plot
plt.title("Price distribution in range 0-1000")
price_cats = pd.cut(data['price'], bins=[0,100,200,300,400,500,600,700,800,900], include_lowest=True)
price_cats.value_counts(sort=False).plot(kind='bar', figsize=(6,4))
plt.savefig('output/fig7.png', bbox_inches='tight')

plt.clf() # Reset plot
plt.title("Correlation of price with other numeric parameters")
plt.xlabel("Parameters")
plt.ylabel("Correlation coefficient")
data2 = data.copy()
data2 = data2.replace('Private room', 0)
data2 = data2.replace('Entire home/apt', 1)
data2 = data2.replace('Shared room', 2)
data2.drop(columns=['price', 'latitude', 'longitude']).corrwith(data2['price']).plot(figsize=(6,4), kind='bar')
plt.savefig('output/fig8.png', bbox_inches='tight')

plt.clf() # Reset plot
# Display the correlations between all parameters
rcParams['figure.figsize'] = 14,10
sns.heatmap(data2.corr(),annot=False, fmt='g')
plt.savefig('output/fig9.png', bbox_inches='tight')

plt.clf() # Reset plot
# Correlations between the price and other numeric parameters
sns.pairplot(data=data2, x_vars=['host_id', 'latitude', 'longitude', 'minimum_nights', 'room_type', 'availability_365'], y_vars=['price'], kind='scatter')
plt.savefig('output/fig10.png', bbox_inches='tight')

# Create the bounding box (tuple) for the map plot
box = (round(data.longitude.min(),4), round(data.longitude.max(),4), round(data.latitude.min(),4), round(data.latitude.max(),4))

plt.clf() # Reset plot
rcParams['figure.figsize'] = 14,10

# Create scatterplots for each room type on the map of NYC
fig, ax = plt.subplots()
ax.scatter(data[data.room_type=="Private room"].longitude, 
           data[data.room_type=="Private room"].latitude, 
           alpha= 0.7, c='tab:blue', s=12)

# Less opacity due to it being way too dense and bright
ax.scatter(data[data.room_type=="Entire home/apt"].longitude, 
           data[data.room_type=="Entire home/apt"].latitude, 
           alpha= 0.25, c='tab:orange', s=12)

ax.scatter(data[data.room_type=="Shared room"].longitude, 
           data[data.room_type=="Shared room"].latitude, 
           alpha= 0.7, c='tab:green', s=12)

# Limit the plots by the size of the map
ax.set_xlim(box[0],box[1])
ax.set_ylim(box[2],box[3])

# Update the legend opacity and size for better visibility
legend = ax.legend(['Private room', 'Entire home/apt', 'Shared room'])
for lh in legend.legendHandles: 
    lh.set_alpha(1)
    lh.set_sizes([30])
    
plt.title("Distribution of Airbnb listings by room type")
plt.xlabel("Latitude")
plt.ylabel("Longitude")
    
ax.imshow(plt.imread("data/NYC.jpg"), zorder=0, extent = box, aspect='equal')
plt.savefig('output/fig11.png', bbox_inches='tight')
# =========================================================


# === Models ==============================================
print("\nSplitting the data into training and testing data...")
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['price', 'name', 'host_name']), 
                                                    data['price'], test_size = 0.30, random_state = 0)
                                                    
# Connect training and testing data into a single dataframe
train = X_train.copy()
train['source'] = 'train'
test = X_test.copy()
test['source'] = 'test'
data3 = pd.concat([train, test])

# Use one-hot encoding to transform non-binary data into binary data
print("One-hot encoding data...")
data_dum = pd.get_dummies(data3, columns=data3.drop(columns=['source']).select_dtypes(object).columns)

# Split the data into training and testing partitions once again
train = data_dum[data_dum.source == 'train'].drop(columns=['source'])
test = data_dum[data_dum.source == 'test'].drop(columns=['source'])

# Create a dataframe for saving the results of the models
df = pd.DataFrame({'id': test.index, 'actual_price': np.array(y_test)})

# Linear regression model
print("\nRunning a linear regression model...")
reg = LinearRegression()
reg_fit = reg.fit(train, y_train)
pred_linreg = reg_fit.predict(test)
df['price_linreg'] = pred_linreg

# RandomForest model
print("Running a RandomForest model...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
rf_fit = rf.fit(train, y_train)
pred_rf = rf.predict(test)
df['price_rf'] = pred_rf

print("RF accuracy:", round(accuracy_score(df.actual_price, df.price_rf)*100,2), "%")

# KNN models
knn3 = KNeighborsClassifier(n_neighbors=3)
knn3.fit(train, y_train)
pred_knn3 = knn3.predict(test)
df['price_knn3'] = pred_knn3

knn30 = KNeighborsClassifier(n_neighbors=30)
knn30.fit(train, y_train)
pred_knn30 = knn30.predict(test)
df['price_knn30'] = pred_knn30

knn100 = KNeighborsClassifier(n_neighbors=100)
knn100.fit(train, y_train)
pred_knn100 = knn100.predict(test)
df['price_knn100'] = pred_knn100

print("KNN3 Accuracy:", accuracy_score(df.actual_price, df.price_knn3)*100, "%")
print("KNN30 Accuracy:", accuracy_score(df.actual_price, df.price_knn30)*100, "%")
print("KNN100 Accuracy:", accuracy_score(df.actual_price, df.price_knn100)*100, "%")

# Model visualizations
plt.clf() # Reset plot
print("\nCreating visualizations for the models...")
rcParams['figure.figsize'] = 10,8
plt.title("Price predictions")
sns.regplot(y=df.actual_price,x=df.price_linreg)
sns.regplot(y=df.actual_price,x=df.price_rf)
sns.regplot(y=df.actual_price,x=df.price_knn30)
sns.regplot(y=df.actual_price,x=df.price_knn100)
plt.plot()
plt.legend(['LinReg', 'RF', 'KNN30', 'KNN100'])
plt.ylabel("Actual price")
plt.xlabel("Predicted price")
plt.savefig('output/fig12.png', bbox_inches='tight')
# =========================================================
