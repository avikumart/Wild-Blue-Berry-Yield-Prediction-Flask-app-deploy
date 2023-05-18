# Wild-Blue-Berry-Yield-Prediction
In this repo, I have worked on Wild Blue Berry Yield prediction project. Dataset was generated by Pollination Simulation Model for research purpose

**Description:** 
The dataset used for predictive modelling was generated by the Wild Blueberry Pollination Simulation Model, which is an open-source, spatially-explicit computer simulation program, that enables exploration of how various factors, including plant spatial arrangement, outcrossing and self-pollination, bee species compositions and weather conditions, in isolation and combination, affect pollination efficiency and yield of the wild blueberry agro-ecosystem. The simulation model has been validated by the field observation and experimental data collected in Maine USA and Canadian Maritimes during the last 30 years and now is a useful tool for hypothesis testing and theory development for wild blueberry pollination researches. This simulated data provides researchers who have actual data collected from field observation and those who wants to experiment the potential of machine learning algorithms response to real data and computer simulation modelling generated data as input for crop yield prediction models.

**Problem Statement:**
The target feature is `yield` which is a `continuous variable`. The task is to classify this variable based on the other 17 features step-by-step by going through each day's task. The evaluation metrics will be `RMSE` score

### Web App: [Click Here](https://wbb-prediction.onrender.com/)

**Solution:**

1) EDA using matplotlib, pandas and seaborn
2) Feature selection using `mutual_info_regressor`
3) Kmeans Clustering to cluster types of bees columns
4) Standardizing input features
5) Basline modeling using gradient boosted trees: `RMSE - 188`
6) Cross validation using gradient boosted trees: `RMSE - 141`
7) Model hyperparameters tuning using pipeline object with XGBRegressor: `RMSE - 18`
8) Explainable AI using `shap` 

**Learning outcomes:**

- Feature selection methods
- Development of machine learning pipelines using sklearn's pipeline object

Acknowledegement: ***TMLC Academy*** 
