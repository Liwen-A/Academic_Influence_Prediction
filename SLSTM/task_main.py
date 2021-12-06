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
parser.add_argument('--nhid_fc', type=int, default=20,
                    help='hidden size of FC layers')
parser.add_argument('--nhid_abs', type=int, default=500,
                    help='hidden size of FC layers for abstract')
parser.add_argument('--nhid', type=int, default=40,
                    help='hidden size of recurrent net')
parser.add_argument('--epochs', type=int, default=5000,
                    help='number of epochs')
parser.add_argument('--batch', type=int, default=50,
                    help='batch size')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--calc_no', default=1000,
                    help='Calculation iteration number of LSTM')

args = parser.parse_args()

#a_lr = np.logspace(-4,-2,num=1000, base=10)
#args.lr = round(float(a_lr[int(np.random.randint(1000))]),5)

print(args)


ninp_fc = 2
nout_fc = int(0.5*args.nhid)
ninp_abs = 300
ninp = 1
nout = 1
batch_test = args.batch

trainloader, train_dataset, validloader, valid_dataset, testloader, test_dataset = utils.get_data(args.batch,batch_test)
model = network.oLSTM(ninp_fc, args.nhid_fc, nout_fc, ninp_abs, args.nhid_abs, ninp, args.nhid, args.nlayers, args.batch, nout)

objective = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
objective_test = nn.MSELoss(reduction='sum')

def r2_loss(output, label):
    label_mean = torch.mean(label)
    ss_tot = torch.sum((label - label_mean) ** 2)
    ss_res = torch.sum((label - output) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

def test(dataloader,dataset):
    model.eval()
    loss = 0.
    r2_coeff = 0.
    with torch.no_grad():
        for i, (data_nts, data_abs, data_ts, label) in enumerate(dataloader):
            if i==0:
                label_all  = label
            data_ts = data_ts.permute(1, 0, 2)		
            output = model(data_nts, data_abs, data_ts).squeeze(-1)
            if i==0:
                output_all  = output
            loss += objective_test(output, label)
            if i!=0:
                label_all  = torch.cat((label_all, label), 0)
                output_all = torch.cat((output_all, output), 0)
        loss /= len(dataset)
        r2_coeff = r2_loss(output_all, label_all)
        loss = torch.sqrt(loss)
    return (loss.item(), r2_coeff.item())

best_val_loss = 100000.

start_time = time.time()

for epoch in range(args.epochs):
    model.train()
    for i, (data_nts, data_abs, data_ts, label) in enumerate(trainloader):
        optimizer.zero_grad()
        data_ts = data_ts.permute(1, 0, 2)
        output = model(data_nts, data_abs, data_ts).squeeze(-1)
#        if (epoch%500 == 0.0):
#            Path('results').mkdir(parents=True, exist_ok=True)
#            f = open('results/' + 'log0_'+ str(args.calc_no) + '.txt', 'a')
#            f.write('pred: ' + str(output) + '\n')    
#            f.write('label: ' + str(label) + '\n')
#            f.close()
        loss = objective(output, label)
        loss.backward()
        optimizer.step()

    train_loss, train_r2 = test(trainloader,train_dataset)
    valid_loss, valid_r2 = test(validloader,valid_dataset)
    test_loss, test_r2 = test(testloader,test_dataset)
#    if(valid_loss<best_val_loss):
#        best_val_loss = valid_loss
#        best_eval_test_loss = test_loss
#    final_test_loss = test_loss

#    if (epoch == 0):
#        f.write('## learning rate = ' + str(args.lr) + '\n')
    if (epoch%50 == 0.0):
        Path('results').mkdir(parents=True, exist_ok=True)
        f1 = open('results/' + 'train_RMSE' + '.txt', 'a')
        f1.write(str(round(train_loss, 2)) + '\n')
        f1.close()
        f2 = open('results/' + 'eval_RMSE' + '.txt', 'a')
        f2.write(str(round(valid_loss, 2)) + '\n')
        f2.close()
        f3 = open('results/' + 'test_RMSE' + '.txt', 'a')
        f3.write(str(round(test_loss, 2)) + '\n')
        f3.close()
        f4 = open('results/' + 'train_r2' + '.txt', 'a')
        f4.write(str(round(train_r2, 2)) + '\n')
        f4.close()
        f5 = open('results/' + 'eval_r2' + '.txt', 'a')
        f5.write(str(round(valid_r2, 2)) + '\n')
        f5.close()
        f6 = open('results/' + 'test_r2' + '.txt', 'a')
        f6.write(str(round(test_r2, 2)) + '\n')
        f6.close()

    if (epoch%100) == 0.0:
        args.lr /= 2.
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr

end_time = time.time()

#f = open('results/'+ 'log_'+ str(args.calc_no) + '.txt', 'a')
#f.write('best eval loss: ' + str(round(best_val_loss, 2)) + '\n')
#f.write('best eval test loss: ' + str(round(best_eval_test_loss, 2)) + '\n')
#f.write('execution time (all epochs): ' + str(round((end_time - start_time)/3600.0, 3)) + 'hours' + '\n')
#f.close()
