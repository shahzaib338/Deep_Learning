import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
df = pd.read_csv(url)
print(df.head())
print(df.shape)
print(df.columns.tolist())
print(df.isnull().sum())
X = df.drop("target", axis=1)
y = df["target"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(X_train.shape)

class CancerDetector(nn.Module):
    def __init__(self):
        super(CancerDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(13, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()

        )
    def forward(self, x):
        return self.model(x)


model = CancerDetector()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(10):
    for batch_X, batch_y in dataloader:
        prediction = model(batch_X)
        loss = loss_fn(prediction.squeeze(), batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss {loss.item():.4f}")


X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test.values.copy())


model.eval()


with torch.no_grad():
    predictions = model(X_test)          
    predictions = (predictions.squeeze() > 0.5).float()  
    accuracy = (predictions == y_test).float().mean()    
    print(f"Accuracy: {accuracy.item()*100:.2f}%")


Check = df.drop("target", axis=1)
values = []


while True:
    
    try: 
        for coloum in Check.columns:
            value = float(input(f"Enter {coloum}: "))
            values.append(value)
        model.eval()
        new_patient = np.array(values).reshape(1,-1)
        new_patient_scaled = scaler.transform(new_patient)
        new_patient_tensor = torch.FloatTensor(new_patient_scaled)

        with torch.no_grad():
            prediction = model(new_patient_tensor)
            result  = "Heart Disease Detected " if prediction.item() >0.5 else "No Heart Disease"
            print(result)
        
        again = input("Do you want to check Another Patient (y/n)")
        if again.lower() == "y":
            values = []
            continue
        else:
            print("Good Bye")
            break
            

    except ValueError:
        print("Please Enter Correct Values")
    except Exception as e:
        print("Exception ", e )
