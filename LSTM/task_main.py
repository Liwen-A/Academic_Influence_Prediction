from torch import nn, optim
import utils
import network
import torch
from pathlib import Path
import argparse
import numpy as np
import time

parser = argparse.ArgumentParser(description='training parameters')

#parser.add_argument('--names', default=['Aeronauticsdata'],
#                    help='which datasets (.csv filenames) to include')
parser.add_argument('--nhid', type=int, default=40,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=500,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=50,
                    help='batch size')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--calc_no', default=1000,
                    help='Calculation iteration number of LSTM')

args = parser.parse_args()

a_lr = np.logspace(-4,-2,num=1000, base=10)
args.lr = round(float(a_lr[int(np.random.randint(1000))]),5)

print(args)

ninp = 1
nout = 1
batch_test = 50

trainloader, train_dataset, validloader, valid_dataset, testloader, test_dataset = utils.get_data(args.batch,batch_test)
model = network.oLSTM(ninp, args.nhid, args.nlayers, nout)

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
objective_test = nn.MSELoss(reduction='sum')

def test(dataloader,dataset):
    model.eval()
    loss = 0.
    with torch.no_grad():
        for i, (data, label) in enumerate(dataloader):
            data = data.permute(1, 0, 2)		
            output = model(data).squeeze(-1)
            loss += objective_test(output, label)
        loss /= len(dataset)
        loss = torch.sqrt(loss)
    return loss.item()

best_val_loss = 100000.

start_time = time.time()

for epoch in range(args.epochs):
    model.train()
    for i, (data, label) in enumerate(trainloader):
        optimizer.zero_grad()
        data = data.permute(1, 0, 2)
        output = model(data).squeeze(-1)
        loss = objective(output, label)
        loss.backward()
        optimizer.step()

    train_loss = test(trainloader,train_dataset)
    valid_loss = test(validloader,valid_dataset)
    test_loss = test(testloader,test_dataset)
    if(valid_loss<best_val_loss):
        best_val_loss = valid_loss
        best_eval_test_loss = test_loss
#    final_test_loss = test_loss

    Path('results').mkdir(parents=True, exist_ok=True)
    f = open('results/' + 'log_'+ str(args.calc_no) + '.txt', 'a')
    if (epoch == 0):
        f.write('## learning rate = ' + str(args.lr) + '\n')
    f.write('train loss: ' + str(round(train_loss, 2)) + '\n')
    f.write('eval loss: ' + str(round(valid_loss, 2)) + '\n')
    f.write('test loss: ' + str(round(test_loss, 2)) + '\n')
    f.close()

    if (epoch + 1) == 250:
        args.lr /= 10.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

end_time = time.time()

f = open('results/'+ 'log_'+ str(args.calc_no) + '.txt', 'a')
f.write('best eval loss: ' + str(round(best_val_loss, 2)) + '\n')
f.write('best eval test loss: ' + str(round(best_eval_test_loss, 2)) + '\n')
f.write('execution time (all epochs): ' + str(round((end_time - start_time)/3600.0, 3)) + 'hours' + '\n')
f.close()
