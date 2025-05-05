import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_c
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()

# See how many survived
women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)
print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)
print("% of men who survived:", rate_men)


y = train_data["Survived"]

nan_count_Pclass = train_data['Pclass'].isna().sum()
nan_count_Sex = train_data["Sex"].isna().sum()
nan_count_Age = train_data["Age"].isna().sum()
nan_count_SibSp = train_data["SibSp"].isna().sum()
nan_count_Parch = train_data["Parch"].isna().sum()
nan_count_Fare = train_data["Fare"].isna().sum()
nan_count_Embarked = train_data["Embarked"].isna().sum()
print("nan count Pclass:", nan_count_Pclass)
print("nan count Sex:", nan_count_Sex)
print("nan count Age:", nan_count_Age)
print("% of Age NaN:", nan_count_Age / len(train_data))
print("nan count SibSp:", nan_count_SibSp)
print("nan count Parch:", nan_count_Parch)
print("nan count Fare:", nan_count_Fare)
print("nan count Embarked:", nan_count_Embarked)

nan_count_Pclass_test = test_data['Pclass'].isna().sum()
nan_count_Sex_test = test_data["Sex"].isna().sum()
nan_count_Age_test = test_data["Age"].isna().sum()
nan_count_SibSp_test = test_data["SibSp"].isna().sum()
nan_count_Parch_test = test_data["Parch"].isna().sum()
nan_count_Fare_test = test_data["Fare"].isna().sum()
nan_count_Embarked_test = test_data["Embarked"].isna().sum()
print("nan count Pclass test:", nan_count_Pclass_test)
print("nan count Sex test:", nan_count_Sex_test)
print("nan count Age test:", nan_count_Age_test)
print("% of Age NaN test:", nan_count_Age_test / len(test_data))
print("nan count SibSp test:", nan_count_SibSp_test)
print("nan count Parch test:", nan_count_Parch_test)
print("nan count Fare test:", nan_count_Fare_test)
print("nan count Embarked test:", nan_count_Embarked_test)


features = ["Pclass", "Sex", "SibSp", "Parch", "Fare", "Embarked"]


train_data["Embarked"].fillna(train_data["Embarked"].mode().iloc[0], inplace=True)
test_data["Fare"].fillna(test_data["Fare"].mean(), inplace=True)

X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])
model = GaussianNB()
#model = LinearSVC(random_state=0)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)
print("Your submission was successfully saved!")