# Cricket Match Win Percentage Prediction - README
## 1. Goal
The goal of this project is to develop a machine learning model using Logistic Regression and Random Forest Classifier in Python to predict the win percentage of a cricket team at any instant during an Indian Premier League (IPL) match. The model leverages historical IPL data to estimate the likelihood of a team winning based on match conditions, current score, overs bowled, wickets lost, and other relevant factors.

## 2. Process
### 2.1. Tools and Libraries
#### Python: Programming language used for the model development.
#### Pandas: For data manipulation and analysis.
#### NumPy: For numerical computations.
#### Scikit-learn: For building and evaluating machine learning models.
#### Matplotlib/Seaborn: For data visualization.
### 2.2. Data Collection and Preprocessing
#### IPL Dataset:

The dataset contains historical IPL match data, including match details like teams, scores, overs, wickets, and outcomes.\
The dataset includes features like runs scored, wickets lost, overs bowled, run rate, required run rate, etc., at various points in the match.\
#### Feature Engineering:

Relevant features were extracted and created to improve model predictions. Some key features include:\
Current Score: Runs scored at any instant.\
Wickets Lost: Number of wickets lost by the batting team.\
Overs Bowled: Number of overs completed.\
Run Rate: Current and required run rates.\
Target Variable: Win/Loss status of the team.\
#### Data Splitting:

The dataset was split into training and testing sets (e.g., 80% training, 20% testing) to evaluate model performance.\
### 2.3. Model Development
#### Logistic Regression:

A binary classification model was developed using logistic regression to predict the win/loss probability based on match conditions.\
The model provides a probability score that represents the likelihood of winning.\
#### Random Forest Classifier:

A more complex model using a Random Forest Classifier was built to capture non-linear relationships between features.\
The Random Forest model aggregates multiple decision trees to improve prediction accuracy and robustness.\
#### Model Training and Evaluation:

Both models were trained using the training data and evaluated using accuracy, precision, recall, F1-score, and ROC-AUC metrics.\
Cross-validation techniques were applied to ensure model generalization and avoid overfitting.
### 2.4. Hyperparameter Tuning
Grid search and randomized search were used to fine-tune model parameters like the number of trees in the Random Forest, depth of trees, and regularization strength for Logistic Regression.
### 2.5. Model Deployment
The final models were saved using joblib for easy deployment and usage in real-time predictions.
## 3. Result
The developed models can predict the win percentage of a cricket team at any given moment during an IPL match with reasonable accuracy. Below are the key outcomes:\

### Model Performance:
#### Logistic Regression:

Simpler model with moderate accuracy and quick inference.\
Best suited for linear relationships between features.\
Provides interpretable win probabilities.
#### Random Forest Classifier:

Achieved higher accuracy by capturing complex patterns in the data.\
Handles feature importance effectively and is robust to overfitting.\
Suitable for real-time predictions with slightly higher computational cost.
### Key Insights:
The win percentage is heavily influenced by features like the number of wickets lost, the required run rate, and the stage of the match (early vs. late overs).\
The Random Forest model was particularly effective at predicting outcomes in close matches.
### Conclusion:
The project demonstrates how machine learning can be applied to predict the outcome of sports events, leveraging historical data and in-match statistics. The models provide real-time win probabilities, which can be useful for broadcasters, analysts, and fans to gauge the momentum of a cricket match as it unfolds.

### Future Improvements:
Incorporate additional features like player performance, venue conditions, and opposition team strengths.\
Explore deep learning models for more complex patterns.\
Integrate the model into a live dashboard for real-time win percentage updates during matches.
