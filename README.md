# Predictive Modeling & Analytics on Home Equity Line of Credit Data (Python)

## HMEQ Data Set
In this assignment we will use Python to examine a data set containing Home Equity Loans. The data set contains two target variables. The first target, TARGET_BAD_FLAG indicates whether or not the loan defaulted. If the value is set to 1, then the loan went bad and the bank lost money. If the value is set to 0, the loan was repaid.

The second target, TARGET_LOSS_AMT, indicates the amount of money that was lost for loans that went bad. The remaining variables contain information about the customer at the time that the loan was issued.

This is the data that we will use throughout this class in order to develop predictive models that will be used to determine the level of risk for each loan.

As with all real world data, this data is far from perfect.

It contains both numerical and categorical variables.
It contains missing data.
It contains outliers.

## Table of Contents
- [Data Preparation](#heading)
- [Tree Based Models](#heading-1)
- [Regression Based Models](#heading-2)
- [Neural Network](#heading-3)


# Building Machine Learning Models

> Developed different predictive models to determine the level risk of each loan based on whether or not loans defaulted, and loss amount on bad loans. Evaluated each model with ROC curve and RMSE accuracy metrics. 

<!-- toc -->

## Data Preparation

- Download the HMEQ Data set
- Read the data into Python
- Explore both the input and target variables using statistical techniques.
- Explore both the input and target variables using graphs and other visualization.
- Look for relationships between the input variables and the targets.
- Fix (impute) all missing data.
- Note: For numerical data, create a flag variable to indicate if the value was missing
- Convert all categorical variables numeric variables

## Tree Based Models

We will continue to use Python to develop predictive models. In this assignment, we will use three different tree based techniques to analyze the data: DECISION TREES, RANDOM FORESTS, and GRADIENT BOOSTING. The deliverables for each technique are given below.

### Create a Training and Test Data Set:

### Decision Trees:
- Develop a decision tree to predict the probability of default
- Calculate the accuracy of the model on both the training and test data set
- Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
- Display the Decision Tree using a Graphviz program
- List the variables included in the decision tree that predict loan default.
- Develop a decision tree to predict the loss amount assuming that the loan defaults
- Calculate the RMSE for both the training data set and the test data set
- Display the Decision Tree using a Graphviz program
- List the variables included in the decision tree that predict loss amount.

### Random Forests:
- Develop a Random Forest to predict the probability of default
- Calculate the accuracy of the model on both the training and test data set
- Create a graph that shows the ROC curves for both the training and test data set. Clearly label each curve and display the Area Under the ROC curve.
- List the variables included in the Random Forest that predict loan default.
- Develop a Random Forest to predict the loss amount assuming that the loan defaults
- Calculate the RMSE for both the training data set and the test data set
- List the variables included in the Random Forest that predict loss amount.

### Gradient Boosting:
- Develop a Gradient Boosting model to predict the probability of default
- Calculate the accuracy of the model on both the training and test data set
- Create a graph that shows the ROC curves for both the training and test data set. Clearly - label each curve and display the Area Under the ROC curve.
- List the variables included in the Gradient Boosting that predict loan default.
- Develop a Gradient Boosting to predict the loss amount assuming that the loan defaults
- Calculate the RMSE for both the training data set and the test data set
- List the variables included in the Gradient Boosting that predict loss amount.

### ROC Curves:
- Generate a ROC curve for the Decision Tree, Random Forest, and Gradient Boosting models using the Test Data Set
- Use different colors for each curve and clearly label them
- Include the Area under the ROC Curve (AUC) on the graph.

## Heading

This is an h1 heading

## Heading

This is an h1 heading



## Data Dictionary

| VARIABLE          | DEFINITION                                                                                                                                                             | ROLE   | TYPE     | CONVENTIONAL WISDOM                                                                                                                                                                                                                                                                                                    |
| ----------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| TARGET\_BAD\_FLAG | BAD=1 (Loan was defaulted)                                                                                                                                             | TARGET | BINARY   | HMEQ = Home Equity Line of Credit Loan. BINARY TARGET                                                                                                                                                                                                                                                                  |
| TARGET\_LOSS\_AMT | If loan was Bad, this was the amount not repaid.                                                                                                                       | TARGET | NUMBER   | HMEQ = Home Equity Line of Credit Loan. NUMERICAL TARGET                                                                                                                                                                                                                                                               |
| LOAN              | HMEQ Credit Line                                                                                                                                                       | INPUT  | NUMBER   | The bigger the loan, the more risky the person                                                                                                                                                                                                                                                                         |
| MORTDUE           | Current Outstanding Mortgage Balance                                                                                                                                   | INPUT  | NUMBER   | If you owe a lot of money on your current mortgage versus the value of your house, you are more risky.                                                                                                                                                                                                                 |
| VALUE             | Value of your house                                                                                                                                                    | INPUT  | NUMBER   | If you owe a lot of money on your current mortgage versus the value of your house, you are more risky.                                                                                                                                                                                                                 |
| REASON            | Why do you want a loan?                                                                                                                                                | INPUT  | CATEGORY | If you are consolidating debt, that might mean you are having financial trouble.                                                                                                                                                                                                                                       |
| JOB               | What do you do for a living?                                                                                                                                           | INPUT  | CATEGORY | Some jobs are unstable (and therefore are more risky)                                                                                                                                                                                                                                                                  |
| YOJ               | Years on Job                                                                                                                                                           | INPUT  | NUMBER   | If you habe been at your job for a while, you are less likely to lose that job. That makes you less risky.                                                                                                                                                                                                             |
| DEROG             | Derogatory Marks on Credit Record. These are very bad things that stay on your credit report for 7 years. These include bankruptcies or leins placed on your property. | INPUT  | NUMBER   | Lots of Derogatories mean that something really bad happened to you (such as a bankruptcy) in your past. This makes you more risky.                                                                                                                                                                                    |
| DELINQ            | Delinquencies on your current credit report. This refers to the number of times you were overdue when paying bills in the last three years.                            | INPUT  | NUMBER   | When you have a lot of delinquencies, you might be more likely to default on a loan.                                                                                                                                                                                                                                   |
| CLAGE             | Credit Line Age (in months) is how long you have had credit. Are you a new high school student with a new credit card or have you had credit cards for many years?     | INPUT  | NUMBER   | If you have had credit for a long time, you are considered less risky than a new high school student.                                                                                                                                                                                                                  |
| NINQ              | Number of inquiries. This is the number of times within the last 3 years that you went out looking for credit (such as opening a credit card at a store)               | INPUT  | NUMBER   | Conventional wisdom in that if you are looking for more credit, you might be in financial trouble. Thus you are risky.                                                                                                                                                                                                 |
| CLNO              | Number of credit lines you have (credit cards, loans, etc.).                                                                                                           | INPUT  | NUMBER   | This is a double edged swoard. Peole who have a lot of credit lines tend to be safe. The reason is that if OTHER PEOPLE think you are trustworthy enough for a credit card, then maybe you are. However, if you have too many credit lines, you might be risky because you have the potential to run up a lot of debt. |
| DEBTINC           | Debt to Income Ratio. Take the money you spend every month and divide it by the amount of money you earn every month.                                                  | INPUT  | NUMBER   | If your debt to income ratio is high then you are risky because you might not be able to pay your bills.                                                                                                                                                                                                               |
