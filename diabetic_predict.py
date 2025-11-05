import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'C:\Python\ML\Diabetes prediction\diabetes.csv')
print("First rows from the dataset :- \n", data.head())
print("\nISNULL sum for the dataset :- \n", data.isnull().sum())

x = data.drop(columns='Outcome', axis=1)
y = data['Outcome']

sc = StandardScaler()
sc.fit(x)
x_scaled = sc.transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_scaled, y, test_size=0.1, stratify=y, random_state=1
)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)

predict = model.predict(x_train)
acc = accuracy_score(y_train, predict)
print("\nTraining Accuracy: {:.2f}%".format(acc * 100))

test_predictions = model.predict(x_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print("Testing Accuracy: {:.2f}%".format(test_accuracy * 100))

random_index = np.random.randint(0, x_test.shape[0])
sample_data = np.array(x_test[random_index]).reshape(1, -1)
true_label = y_test.iloc[random_index]
print("\nThe random data selected from the dataset :- \n",sample_data)
prediction = model.predict(sample_data)

print("\n--- Random Sample Prediction ---")
print("Predicted (0 = Non-Diabetic, 1 = Diabetic):", prediction[0])
print("Actual Label:", true_label)

if prediction[0] == 1:
    print("\n-- Result: The person is Diabetic --")
else:
    print("\n-- Result: The person is Non-Diabetic --")
