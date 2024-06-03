## Drop out rate prediction for high school students
## Project Overview

This project focuses on visualizing and analyzing student data to identify patterns and insights related to student enrollment, dropout rates, and graduation rates. Various data visualization techniques and machine learning models are utilized to achieve these objectives.

## Table of Contents

1. [Data Visualization](#data-visualization)
2. [Modeling and Prediction](#modeling-and-prediction)
3. [Creating a System for Prediction](#creating-a-system-for-prediction)

## Data Visualization

The project starts with an exploration of the student dataset, using different visualization techniques to understand the data distribution and relationships.

### Gender Distribution

```python
sns.countplot(data=student, x='Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Students')
plt.show()
```
The gender distribution is visualized to understand the balance between male and female students.

### Nationality Distribution

``` python
sns.countplot(data=student, x='Nationality')
plt.xticks(rotation=90)
plt.xlabel('Nationality')
plt.ylabel('Number of Students')
plt.show()
```
The nationality distribution helps identify the most common nationalities among the students.

### Displaced Students

```python
sns.countplot(data=student, x='Displaced', hue='Target', hue_order=['Dropout', 'Enrolled', 'Graduate'])
plt.xticks(ticks=[0,1], labels=['No','Yes'])
plt.ylabel('Number of Students')
plt.show()
```
This plot shows the number of displaced students and their corresponding enrollment status.

### International Students

```python

sns.countplot(data=student, x='International', hue='Target', hue_order=['Dropout', 'Enrolled', 'Graduate'])
plt.xticks(ticks=[0,1], labels=['No','Yes'])
plt.ylabel('Number of Students')
plt.show()
```
The distribution of international students and their enrollment status is analyzed.

### Modeling and Prediction

Different machine learning models are implemented to predict student outcomes based on various features.
Logistic Regression

Logistic regression is used to model the probability of different outcomes.

```python

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)
y_pred = log_reg.predict(X_test)
```
### Random Forest

A random forest classifier is implemented to improve prediction accuracy.

```python

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)
y_pred_rf = rf.predict(X_test)
```
### Model Evaluation

The performance of different models is evaluated using metrics such as accuracy and ROC curves.

```python

from sklearn.metrics import accuracy_score, RocCurveDisplay
accuracy = accuracy_score(Y_test, y_pred)
print("Accuracy:", accuracy)

RocCurveDisplay.from_predictions(Y_test, y_pred_rf)
plt.show()
```
### Creating a System for Prediction

A system is created to predict student outcomes based on input data.

This system uses the trained model to make predictions based on new data inputs. 
