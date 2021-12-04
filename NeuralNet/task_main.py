# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 01:13:49 2021

@author: Shreya
"""
from torch import nn, optim
import utils
import network
import torch
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description='training parameters')

parser.add_argument('--names', default=['MedNobel', 'PhysicsNobel', 'ChemNobel'],
                    help='which datasets (.csv filenames) to include')
parser.add_argument('--nhid1', type=int, default=50,
                    help='hidden size of layer 1 of the neural network')
parser.add_argument('--nhid2', type=int, default=20,
                    help='hidden size of layer 2 of the neural network')
parser.add_argument('--nhid3', type=int, default=5,
                    help='hidden size of layer 3 of the neural network')
parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=100,
                    help='training batch size')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate')


args = parser.parse_args()
print(args)

ninp = 312                      # 3 input features being used (as of now)
nout = 1
batch_test = 100

trainloader, train_dataset, testloader, test_dataset = utils.get_data(args.names,args.batch,batch_test)
#model = network.NeuralNet(ninp, args.nhid1, args.nhid2, args.nhid3, nout)
model = network.SparseNN()

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
objective_test = nn.MSELoss(reduction='sum')

def test(dataloader,dataset):
    model.eval()
    loss = 0.
    with torch.no_grad():
        for i, (data1, data2, label) in enumerate(dataloader):
            output = model(data1, data2).squeeze(-1)
            loss += objective_test(output, label)
        loss /= len(dataset)
        loss = torch.sqrt(loss)
    return loss.item()

def r2(dataloader,dataset, meanset):
    model.eval()
    ss_tot = 0.
    ss_res = 0.
    target_mean = 0.
    batches = 0.
    with torch.no_grad():
        for i, (data1, data2, label) in enumerate(meanset):
            batches += 1
            target_mean += torch.mean(label)
        target_mean = target_mean/batches
        for i, (data1, data2, label) in enumerate(dataloader):
            output = model(data1, data2).squeeze(-1)
            #target_mean = torch.mean(label)
            target = label
            ss_tot += torch.sum((target - target_mean) ** 2)
            ss_res += torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2.item()

for epoch in range(args.epochs):
    model.train()
    for i, (data1, data2, label) in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(data1, data2).squeeze(-1)
        loss = objective(output, label)
        loss.backward()
        optimizer.step()
    scheduler.step()

    train_loss = test(trainloader,train_dataset)
    test_loss = test(testloader,test_dataset)

    Path('results').mkdir(parents=True, exist_ok=True)
    if (epoch % 50 == 0):
        print(epoch)
        print(train_loss, test_loss)
        trainr2 = r2(trainloader,train_dataset, trainloader)
        testr2 = r2(testloader,test_dataset, testloader)
        print(trainr2, testr2)  
        f = open('results/' + 'train_log.txt', 'a')
    #    if (epoch == 0):
    #        f.write('## learning rate = ' + str(args.lr) + '\n')
        f.write(str(round(train_loss, 3)) + '\n')
        f.close()
        
        f = open('results/' + 'test_log.txt', 'a')
    #    if (epoch == 0):
    #        f.write('## learning rate = ' + str(args.lr) + '\n')
        f.write(str(round(test_loss, 3)) + '\n')
        f.close()

#    if (epoch + 1) == 250:
#        args.lr /= 10.
#        for param_group in optimizer.param_groups:
#            param_group['lr'] = args.lr
trainr2 = r2(trainloader,train_dataset, trainloader)
testr2 = r2(testloader,test_dataset, testloader)
print(trainr2, testr2)