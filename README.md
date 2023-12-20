![screenshots1](images/carimage.jpg?raw=true)

# ML_AutoMobile_Price_Prediction
#### Welcome to ML_AutoMobile_Price_Predictor, an open-source data analysis and machine learning project! This repository hosts a comprehensive analysis of a real-world dataset, showcasing the entire data science pipeline from data exploration to model building. The primary goal is to understand the factors influencing car prices and to prepare the data for machine learning modeling.

## Modules/Libraries
* Scikit-learn
* MatPlotLib
* Statistics
* Numpy
* Pandas
* Seaborn
* UCILREPO
* mpl_toolkits
* Scipy



# Data Overview
#### The dataset contains various features related to automobiles, such as mileage, engine characteristics, and physical dimensions. The initial dataset consisted of 201 rows and 26 columns, including both numerical and categorical data. 
### Description of the columns in the dataset.
##### Price: The selling price of the vehicle.
##### Highway-mpg: The fuel efficiency on the highway, measured in miles per gallon (mpg).
##### City-mpg: The fuel efficiency in the city, also measured in miles per gallon (mpg).
##### Peak-rpm: The maximum revolutions per minute (RPM) of the engine.
##### Horsepower: The power output of the engine, measured in horsepower.
##### Compression-ratio: The ratio of the volume of the combustion chamber from its largest capacity to its smallest capacity.
##### Stroke: The distance traveled by the piston in each cycle within the cylinder, measured in inches or millimeters.
##### Bore: The diameter of each cylinder in the engine.
##### Fuel-system: The method or system of delivering fuel to the engine.
##### Engine-size: The volume of the engine's cylinders, typically measured in cubic centimeters or liters.
##### Num-of-cylinders: The number of cylinders in the vehicle's engine.
##### Engine-type: The type or configuration of the engine.
##### Curb-weight: The weight of the vehicle without occupants or baggage.
##### Height: The height of the vehicle.
##### Width: The width of the vehicle.
##### Length: The length of the vehicle.
##### Wheel-base: The distance between the front and rear wheels of the vehicle.
##### Engine-location: The location of the engine in the vehicle (e.g., front, rear).
##### Drive-wheels: The type of wheel drive system (e.g., front-wheel drive, rear-wheel drive).
##### Body-style: The design and shape of the vehicle's body.
##### Num-of-doors: The number of doors on the vehicle.
##### Aspiration: The type of aspiration used (e.g., standard, turbocharged).
##### Fuel-type: The type of fuel the vehicle uses (e.g., gas, diesel).
##### Make: The manufacturer or brand of the vehicle.
##### Normalized-losses: A relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification, and represents the average loss per car per year.


# Statistical Summary

###### Descriptive statistics provided insights into the distribution, variability, and skewness of numerical features. Key observations were made on the range and spread of features like 'horsepower' and 'engine-size'. Missing values were found in several columns, including 'price', 'peak-rpm', 'horsepower', and others. Strategies for handling missing values included removing rows where 'price' was missing and imputing median values for other numerical columns.
### Outliers:
```python
# Bivariate Analysis - Scatter plots for numerical variables against price
def plot_bivariate(data):
    num_columns = data.select_dtypes(include=['float64', 'int64']).columns
    target = 'price'
    fig, axes = plt.subplots(nrows=len(num_columns), figsize=(10, 3 * len(num_columns)))
    for col, ax in zip(num_columns, axes):
        if col != target:
            sns.scatterplot(x=data[col], y=data[target], ax=ax)
            ax.set_title(f'{col} vs {target}')
    plt.tight_layout()

```

### Features like price and horsepower may have outliers (extremely high or low values compared to the rest). These outliers can significantly influence regression models.
##### The analysis of categorical data through box plots revealed significant variations in 'price' across different categories. For instance, luxury car brands, larger vehicle sizes, and advanced engine types were associated with higher prices.
![screenshots1](images/pricedistrby.png?raw=true)
## Distributions:
```python
# Univariate Analysis - Histograms for each numerical variable
def plot_univariate(data):
    num_columns = data.select_dtypes(include=['float64', 'int64']).columns
    fig, axes = plt.subplots(nrows=len(num_columns), figsize=(10, 3 * len(num_columns)))
    for col, ax in zip(num_columns, axes):
        sns.histplot(data[col], kde=True, ax=ax)
        ax.set_title(f'Histogram of {col}')
    plt.tight_layout()
```

![screenshots1](images/histograms.png?raw=true)

## Correlation with Price:

### Strong Positive Correlations: Features like engine-size, curb-weight, horsepower, and width have strong positive correlations with price. This suggests that cars with larger engines, more horsepower, heavier curb weights, and wider bodies tend to be more expensive.
### Negative Correlations: city-mpg and highway-mpg show negative correlations with price, indicating that more fuel-efficient cars tend to be less expensive.
### Feature Distributions:
#### Variability: Some features exhibit a wide range of values, such as horsepower and engine-size, suggesting a diverse set of vehicles in the dataset, from low-power to high-power engines.
Skewness: Certain features may show skewness (either left or right), which could affect model performance and might need transformation for certain types of analysis.

### Engine Characteristics:
```python    
# 3D Scatter Plot for numerical variables against 'price'
def plot_3d_scatter(data, x_col, y_col, z_col, target_col='price'):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[x_col], data[y_col], data[z_col], c=data[target_col], cmap='coolwarm')
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f'3D Scatter Plot: {x_col}, {y_col}, {z_col} vs {target_col}')
    # Color bar
    plt.colorbar(ax.scatter(data[x_col], data[y_col], data[z_col], c=data[target_col], cmap='coolwarm'), label='price')
    plt.show()
```

### Features related to the engine (like engine-size, horsepower, num-of-cylinders) are particularly influential on the vehicle's price. This aligns with the general understanding that engine specifications are major determinants of a car's performance and cost.
![screenshots1](images/output.png?raw=true)

### Fuel Efficiency:
### city-mpg and highway-mpg could be indicative of a vehicle's fuel efficiency. Their negative correlation with price might reflect market trends where consumers pay a premium for powerful, less fuel-efficient vehicles.

![screenshots1](images/citympghigh.png?raw=true)
### Size Dimensions:

### Dimensions like length, width, and curb-weight are correlated with price, suggesting that larger vehicles or those with more features (hence heavier) are generally more expensive.
![screenshots1](images/priceoverenginesize.png?raw=true)
### Correlation analysis revealed that 'engine-size', 'curb-weight', 'horsepower', and 'width' were strongly positively correlated with 'price'. In contrast, 'city-mpg' and 'highway-mpg' were negatively correlated with 'price', indicating a preference for power over fuel efficiency in higher-priced cars.
![screenshots1](images/correlationwprice.png?raw=true)
![screenshots1](images/heatmap.png?raw=true)

## Potential for Feature Engineering:

The relationships between features like horsepower, engine-size, and mpg metrics can be further explored through feature engineering (e.g., creating ratios or combined features) to potentially unveil more insights.

#### Engine Efficiency (engine_efficiency): Calculated as the ratio of horsepower to engine-size. Higher values indicate more horsepower per unit of engine size.

#### Combined Mileage (combined_mpg): A weighted average of city-mpg and highway-mpg, assuming 60% highway and 40% city driving.

#### Polynomial Feature (engine_size_squared): The square of engine-size. This could help the model capture non-linear relationships between engine size and price.

#### Binning (horsepower_bin): Categorizes horsepower into 'Low', 'Medium', or 'High'. This can be useful for models that work well with categorical data.

#### Interaction Term (engine_cylinders_interaction): The product of engine-size and num-of-cylinders. This feature captures the interaction between these two variables.


```python
# Feature 1: Engine Efficiency
auto_data['engine_efficiency'] = auto_data['horsepower'] / auto_data['engine-size']

# Feature 2: Combined Mileage
# Assuming highway mileage is 60% of the time and city mileage is 40% of the time
auto_data['combined_mpg'] = (auto_data['highway-mpg'] * 0.6 + auto_data['city-mpg'] * 0.4)

# Feature 3: Polynomial Feature (engine-size squared)
auto_data['engine_size_squared'] = auto_data['engine-size'] ** 2

# Feature 4: Binning horsepower into categories
bins = [0, 100, 200, max(auto_data['horsepower'])]
labels = ['Low', 'Medium', 'High']
auto_data['horsepower_bin'] = pd.cut(auto_data['horsepower'], bins=bins, labels=labels)

# Feature 5: Interaction Term (engine-size * num-of-cylinders)
auto_data['engine_cylinders_interaction'] = auto_data['engine-size'] * auto_data['num-of-cylinders']

# Displaying the first few rows of the updated dataframe
auto_data[['engine_efficiency', 'combined_mpg', 'engine_size_squared', 'horsepower_bin', 'engine_cylinders_interaction']].head()

```


### Model - Linear / Ridge Regression With PCA
```python
# Separate the target variable 'price' and features
X = auto_data_encoded.drop('price', axis=1)
y = auto_data_encoded['price']

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the feature data
X_scaled = scaler.fit_transform(X)

# Convert the scaled features back to a DataFrame (optional for visualization)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

X_scaled_df.head()

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Determining the number of components to explain 95% of variance
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1

cumulative_variance_ratio, n_components_95

# Splitting the original scaled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Splitting the PCA-transformed data into training and testing sets
X_train_pca, X_test_pca = train_test_split(X_pca[:, :n_components_95], test_size=0.2, random_state=42)

# Initializing models
linear_model = LinearRegression()
ridge_model = Ridge()

# Training and evaluating models with original features
linear_model.fit(X_train, y_train)
linear_predictions = linear_model.predict(X_test)
linear_rmse = mean_squared_error(y_test, linear_predictions, squared=False)
linear_r2 = r2_score(y_test, linear_predictions)

ridge_model.fit(X_train, y_train)
ridge_predictions = ridge_model.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_predictions, squared=False)
ridge_r2 = r2_score(y_test, ridge_predictions)

# Training and evaluating models with PCA features
linear_model.fit(X_train_pca, y_train)
linear_pca_predictions = linear_model.predict(X_test_pca)
linear_pca_rmse = mean_squared_error(y_test, linear_pca_predictions, squared=False)
linear_pca_r2 = r2_score(y_test, linear_pca_predictions)

ridge_model.fit(X_train_pca, y_train)
ridge_pca_predictions = ridge_model.predict(X_test_pca)
ridge_pca_rmse = mean_squared_error(y_test, ridge_pca_predictions, squared=False)
ridge_pca_r2 = r2_score(y_test, ridge_pca_predictions)

# Results
results = {
    "Linear Regression": {"RMSE": linear_rmse, "R2": linear_r2},
    "Ridge Regression": {"RMSE": ridge_rmse, "R2": ridge_r2},
    "Linear Regression with PCA": {"RMSE": linear_pca_rmse, "R2": linear_pca_r2},
    "Ridge Regression with PCA": {"RMSE": ridge_pca_rmse, "R2": ridge_pca_r2}
}

results
{'Linear Regression': {'RMSE': 2628.314052931367, 'R2': 0.9435372540952293},
 'Ridge Regression': {'RMSE': 2651.2035059706013, 'R2': 0.9425495266203687},
 'Linear Regression with PCA': {'RMSE': 3165.044188439928,
  'R2': 0.9181220357319357},
 'Ridge Regression with PCA': {'RMSE': 3171.130584786848,
  'R2': 0.9178068294537354}}

```
### Observations:
#### Linear Regression performed well on the original features. However, it performed significantly worst with PCA, indicating that reducing features did not helped mitigate these issues.
![screenshots1](images/linearFt.png?raw=true)
## Ridge Regression showed strong performance on both the original and PCA-transformed data, with slightly better results on the original features. This suggests that Ridge Regression's inherent regularization helped handle the multicollinearity in the original data.
## The shrinkage of the coefficients is achieved by penalizing the regression model with a penalty term called L2-norm, which is the sum of the squared coefficients
![screenshots1](images/ridgeregression.png?raw=true)
# Conclusion:
## The choice to use PCA depends on the specific model and the trade-offs between performance and model complexity.
## Ridge Regression seems to be a good choice for this dataset, either with or without PCA, though it performs slightly better without PCA.
## For Linear Regression, PCA usually significantly improves performance and is thus recommended.


