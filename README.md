💊 Drug Na_to_K Prediction with Machine Learning
This project demonstrates the use of various regression algorithms to predict the sodium to potassium ratio (Na_to_K) in patients based on their demographic and medical features such as Age, Sex, Blood Pressure (BP), Cholesterol level, and Drug type. The goal is to train and evaluate models that can accurately predict this ratio, which is clinically relevant for drug prescriptions.

📌 Objective
To build and evaluate machine learning regression models that estimate the Na_to_K ratio based on patient attributes. The project covers data preprocessing, exploratory data analysis, model training, performance evaluation, and prediction on new patient data.

🛠️ Tools & Technologies
Programming Language: Python

Libraries:

pandas

numpy

matplotlib

seaborn

scikit-learn

joblib

_tkinter (used internally by matplotlib for plotting)

Models Used:

Random Forest Regressor

Extra Trees Regressor

Decision Tree Regressor

Linear Regression

Evaluation Metrics:

R² Score (Coefficient of Determination)

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

📊 Dataset
The dataset drug200.csv contains records of 200 patients with the following columns:

Column	Description
Age	Age of the patient
Sex	Gender of patient (M/F)
BP	Blood Pressure category (HIGH, NORMAL, LOW)
Cholesterol	Cholesterol level (HIGH, NORMAL, LOW)
Na_to_K	Sodium to Potassium ratio (target variable)
Drug	Drug type prescribed (drugA, drugB, drugC, drugX, DrugY)

📁 Project Structure
bash
Copy
Edit
drug-na_to_k-prediction/
├── drug200.csv                  # Dataset file
├── drug_na_to_k_prediction.py  # Main ML script (your script)
├── model_joblib_random          # Saved Random Forest model
├── model_joblib_extral          # Saved Extra Trees model
├── model_joblib_decision        # Saved Decision Tree model
├── model_joblib_linear          # Saved Linear Regression model
├── README.md                   # This documentation file
├── requirements.txt            # Python dependencies
🧠 Machine Learning Workflow
1. Exploratory Data Analysis (EDA)
Loaded dataset with pandas.

Visualized categorical feature distributions (Sex, BP, Drug) using seaborn countplots.

Confirmed data integrity and class distribution visually.

2. Data Preprocessing
Converted categorical variables to numeric values for model compatibility:

Sex: M → 1, F → 2

BP: HIGH → 1, NORMAL → 2, LOW → 3

Cholesterol: HIGH → 1, NORMAL → 2, LOW → 3

Drug: drugB → 1, drugA → 2, drugX → 3, drugC → 4, DrugY → 5

Separated features (X) and target variable (y):

X = all columns except Na_to_K

y = Na_to_K

3. Train-Test Split
Split data into training (90%) and testing (10%) sets using train_test_split with random_state=8 for reproducibility.

4. Model Training
Trained these models on the training data:

RandomForestRegressor

ExtraTreesRegressor

DecisionTreeRegressor

LinearRegression

5. Prediction and Evaluation
Predicted Na_to_K values on the test data using each model.

Evaluated models with:

R² Score

Mean Absolute Error (MAE)

Mean Squared Error (MSE)

6. Visualization
Plotted actual vs predicted Na_to_K values (first 11 samples) for all models side-by-side.

7. New Prediction
Predicted Na_to_K for a new patient with these features:

python
Copy
Edit
{'Age': 61, 'Sex': 2, 'BP': 3, 'Cholesterol': 1, 'Drug': 5}
Used the Random Forest model for prediction.

8. Model Saving
Saved all trained models with joblib for future inference.

📈 Model Performance Results (Example Output)
plaintext
Copy
Edit
R² Scores:
Random Forest: 0.97
Extra Trees: 0.97
Decision Tree: 0.95
Linear Regression: 0.84

Mean Absolute Error (MAE):
Random Forest: 0.59
Extra Trees: 0.60
Decision Tree: 0.73
Linear Regression: 1.17

Mean Squared Error (MSE):
Random Forest: 0.65
Extra Trees: 0.67
Decision Tree: 0.90
Linear Regression: 2.47

Na_to_K Prediction for new patient {'Age': 61, 'Sex': 2, 'BP': 3, 'Cholesterol': 1, 'Drug': 5}:
[10.89]  # Example output from Random Forest model
Note: Exact values may vary depending on training.

📉 Visualization
Four subplots display actual vs predicted Na_to_K for the first 11 test samples:

Top-left: Random Forest predictions

Top-right: Extra Trees predictions

Bottom-left: Decision Tree predictions

Bottom-right: Linear Regression predictions

💾 Model Saving Code
python
Copy
Edit
import joblib

joblib.dump(extral, 'model_joblib_extral')
joblib.dump(random, 'model_joblib_random')
joblib.dump(linear, 'model_joblib_linear')
joblib.dump(decision, 'model_joblib_decision')
🔧 How to Run
Ensure you have Python installed (3.7+ recommended).

Install dependencies using pip:

bash
Copy
Edit
pip install pandas numpy matplotlib seaborn scikit-learn joblib
Place drug200.csv and your script (drug_na_to_k_prediction.py) in the same directory.

Run the script:

bash
Copy
Edit
python drug_na_to_k_prediction.py
The script will:

Show countplots for Sex, BP, and Drug

Train all models and print evaluation metrics

Plot actual vs predicted values

Predict Na_to_K for the new sample and print the result

Save models to disk
