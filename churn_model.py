import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Age": [25,45,35,50,23,40,60,48],
    "MonthlySpend": [50,120,80,200,40,150,300,180],
    "TenureMonths": [6,24,12,36,4,30,60,40],
    "Churn": [1,0,0,0,1,0,0,0]
}

df = pd.DataFrame(data)

# Features and label
X = df[["Age","MonthlySpend","TenureMonths"]]
y = df["Churn"]

# Split dataset
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train,y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test,predictions)

print("Model Accuracy:",accuracy)
