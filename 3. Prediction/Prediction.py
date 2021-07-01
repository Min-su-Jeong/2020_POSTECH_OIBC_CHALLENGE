import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.datasets as data
import pandas as pd
import numpy as np

device = 'cpu'
torch.manual_seed(777)

# Setting Hyperparameter
learning_rate = 1e-3
training_epochs = 1000
batch_size = 44

# Setting File Path to Train,Test Data
train_data=pd.read_csv('train.csv',header=None,skiprows=[0], usecols=range(1,8))
test_data=pd.read_csv('test.csv',header=None,skiprows=[0], usecols=range(1,8))

x_train_data=train_data.loc[:,2:8]
y_train_data=train_data[[1]]*0.1
print(x_train_data)
print(y_train_data)

x_train_data=np.array(x_train_data)
y_train_data=np.array(y_train_data)

x_train_data=torch.FloatTensor(x_train_data)
y_train_data=torch.FloatTensor(y_train_data)

train_dataset = torch.utils.data.TensorDataset(x_train_data, y_train_data)
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size, shuffle=True, drop_last=True)

class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)

# Learning Model
linear1 = torch.nn.Linear(6,1024,bias=True)
linear2 = torch.nn.Linear(1024,512,bias=True)
linear3 = torch.nn.Linear(512,256,bias=True)
linear4 = torch.nn.Linear(256,128,bias=True)
linear5= torch.nn.Linear(128,64,bias=True)
linear6= torch.nn.Linear(64,32,bias=True)
linear7= torch.nn.Linear(32,1,bias=True)
mish = Mish()
bn1 = torch.nn.BatchNorm1d(1024)
bn2 = torch.nn.BatchNorm1d(128)

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
torch.nn.init.xavier_uniform_(linear4.weight)
torch.nn.init.xavier_uniform_(linear5.weight)
torch.nn.init.xavier_uniform_(linear6.weight)
torch.nn.init.xavier_uniform_(linear7.weight)

model = torch.nn.Sequential(linear1, bn1, mish,
                            linear2, mish,
                            linear3, mish,
                            linear4, bn2, mish,
                            linear5, mish,
                            linear6, mish,
                            linear7).to(device)

loss = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

total_batch = len(data_loader)
for epoch in range(training_epochs):
    avg_cost = 0
    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        hypothesis = model(X)
        cost = loss(hypothesis, Y)
        cost.backward()
        optimizer.step()
        avg_cost += cost / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning finished')


# Prediction
with torch.no_grad():
  x_test_data=test_data.loc[:,2:8]
  x_test_data=np.array(x_test_data)
  x_test_data=torch.from_numpy(x_test_data).float().to(device)
  prediction = model(x_test_data)

correct_prediction = prediction.cpu().numpy().reshape(-1,1)
correct_prediction = correct_prediction*10
print("=============================Prediction Value=============================")
for i in range(len(correct_prediction)):
  print(correct_prediction[i].item())
