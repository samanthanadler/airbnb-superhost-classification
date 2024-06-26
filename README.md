# Model Selection for Logistic Regression: Airbnb Superhosts

During my time as an AI/ML fellow at Break Through Tech, I worked with a dataset of NYC Airbnb listings, mainly exploring if the host of a given Airbnb is a superhost using relevant features. The Jupyter notebook file attached in this repository was the fifth lab of the program's Machine Learning Foundations curriculum, where we built a complete model using the CRISP-DM machine learning life cycle:

- Business Understanding: I used the attached dataset to determine if an Airbnb host is a superhost.
- Data Understanding and Preparation: Through feature engineering (including one-hot encoding using Pandas to enumerate categorical values), I made the dataset more accessible for a machine learning model.
- Modeling: Using sk-learn's built-in functions, I created a LogisticRegression model using the default regularization value of C = 1.0.
- Evaluation: I found the optimal regularization value for a LogisticRegression model using sk-learn's GridSearchCV function and concluded a regularization of C = 100 would be the best-predicting model. I created visualizations using Seaborn and Matplotlib of the two model's ROC curves and calculated the AUC of the curve whose model selects the 5 best features, resulting in a value of over 0.80.
- Deployment: Using the pickle module, I saved the model that performed the best on the Airbnb dataset. The .pkl file is attached in this repository for future use.
