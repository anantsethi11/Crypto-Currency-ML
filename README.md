# Crypto-Currency-ML
Cryptocurrency Price Prediction Using Random Forest Regression
This project aims to predict the closing price of Bitcoin using a Random Forest Regression model based on historical data. 
Steps and Analysis
1.	Data Loading and Preprocessing
o	The dataset is loaded directly from the provided URL into a Pandas DataFrame.
o	Initial exploration includes checking data information, summary statistics, and handling missing values by dropping rows with NaN values.
o	The 'Date' column is converted to datetime format, and new features 'Year', 'Month', and 'Day' are extracted from it.
o	The 'Date' column is then dropped from the DataFrame.
2.	Exploratory Data Analysis
o	The distribution of the 'Close' price is visualized using a histogram with a kernel density estimate.
o	A correlation matrix is plotted to explore relationships between variables.
3.	Model Building
o	The dataset is split into features (X) and the target variable (y), where 'Close' price serves as y.
o	The data is further split into training (80%) and testing (20%) sets using train_test_split.
o	A Random Forest Regressor is trained on the training data to predict the 'Close' price.
4.	Model Evaluation
o	Predictions are made on the test set using the trained model.
o	Mean Squared Error (MSE) and R-squared metrics are calculated to evaluate model performance on the test data.
o	Actual vs Predicted values are plotted to visually assess model accuracy.
o	Feature importances are visualized to understand which features contribute most to predicting the 'Close' price.
Conclusion
The Random Forest Regression model shows promising results in predicting Bitcoin's closing price based on historical data. The evaluation metrics indicate that the model captures a significant portion of the variance in the 'Close' price. Feature importances suggest that certain derived date features (Year, Month, Day) play key roles in predicting price fluctuations.
This project demonstrates a practical application of machine learning techniques in cryptocurrency price prediction, highlighting the importance of feature engineering and model evaluation in financial forecasting.
