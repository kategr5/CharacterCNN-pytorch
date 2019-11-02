import torch
import torch.nn as nn
import torch.optim as optim
from model import Model
from data_loading import load_data


data=input('Type Dataset Choice, AGNews or 20Newsground:  ')



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
print('device used: ',device)

model = Model(data)
model.to(device)

#define train/test here
train_loader,test_loader=load_data(dataset=data)


loss_crit=nn.CrossEntropyLoss()
#optimizer=optim.SGD(model.parameters(),lr=0.001,momentum=0.9)
optimizer=optim.Adam(model.parameters(),lr=0.0001)

print("Starting training...")
for epoch in range(5):

    running_loss=0.0

    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        inputs,labels=torch.tensor(inputs),torch.tensor(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        y_pred=model(inputs)
        loss = loss_crit(y_pred, labels.long())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # print every 10 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 10))
            running_loss = 0.0

print('Finished training')
print('Starting testing')
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = torch.tensor(inputs), torch.tensor(labels)
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels.long()).sum().item()
print('Accuracy of the network on the test inputs: %d %%' % (100 * correct / total))
