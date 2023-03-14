import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import os
import h5py
import sys
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
root =os.getcwd() + '/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def check_file_exist(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path does not exist")
        sys.exit(-1)
    return

class KeyRecovery(nn.Module):
    def __init__(self):
        super(KeyRecovery, self).__init__()
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3),
            nn.AvgPool1d(2,stride=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3),
            nn.AvgPool1d(2,stride=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3),
            nn.AvgPool1d(2, stride=1),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(1616, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.drop = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.reshape(x, (-1,256))
        x = self.drop(x)
        # x = F.softmax(x, dim=0)
        # x = F.sigmoid(x)
        return x


def partial_correct_accuracy(y_true ,batch_size,y_pred):
    accuracy_num = 0
    for i in range(batch_size):
        if np.argmax(y_pred[i]) == y_true[i]:
            accuracy_num+=1


    return accuracy_num

if __name__ == "__main__":

    NUMBER = 15000

    Traces = np.load("for_training/1m/20k_d1/100avg/nor_traces_maxmin.npy")

    Traces = Traces[:NUMBER].astype(np.float32)

    Traces = Traces[:,[i for i in range(130,240)]]
    Traces = torch.from_numpy(Traces)
    labels = np.load("for_training/1m/20k_d1/100avg/label_0.npy")
    labels = labels[:NUMBER].astype(np.float32)
    labels = torch.from_numpy(labels)
    print(labels.shape)
    model_file = "saved_model.txt"

    epoches = 100
    batch_size = 128
    validation = 0.1

    train_dataset = Data.TensorDataset(Traces, labels)
    train_data, valid_data = Data.random_split(train_dataset, [round((1 - validation) * Traces.shape[0]),
                                                               round(validation * Traces.shape[0])],
                                               generator=torch.Generator().manual_seed(42))

    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    print("data finished")

    model_parameter = {}
    net = KeyRecovery()
    net.to(device)
    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, 10, gamma=1)
    for epoch in range(epoches):
        print("running epoch {}". format((epoch)))
        # training
        accuracy_train = 0
        loss_train = 0.0
        net.train()
        for step, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            x_train = x_train.unsqueeze(1)
            y_train = y_train.to(device)
            output = net(x_train)
            loss = loss_function(output, y_train.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_array = output.cpu().detach().numpy()
            accuracy_train += partial_correct_accuracy(y_train.cpu().numpy(), len(y_train), output_array)
            loss_train += loss.data.cpu().numpy()


        print('Epoch: ', epoch, '|Batch: ', step,
                '|train loss: %.4f' % (loss_train / (13500//batch_size)),
                '|train accuracy: %.4f' % (accuracy_train /13500),
                '|learning rate: %.6f' % optimizer.param_groups[0]['lr'])
        accuracy_train = 0
        loss_train = 0.0



        net.eval()
        loss_valid = 0.0
        accuracy_valid = 0

        for step, (x_valid, y_valid) in enumerate(valid_loader):
            x_valid = x_valid.to(device)
            x_valid = x_valid.unsqueeze(1)
            y_valid = y_valid.to(device)
            output = net(x_valid)
            loss = loss_function(output, y_valid.long())
            output_array = output.cpu().detach().numpy()

            accuracy_valid += partial_correct_accuracy(y_valid.cpu().numpy(), len(y_valid), output_array)
            loss_valid += loss.data.cpu().numpy()

        print('Epoch: ', epoch + 1,
              '|valid loss: %.4f' % (loss_valid / (1500/batch_size)),
              '|valid accuracy: %.4f' % (accuracy_valid / 1500))

        scheduler.step()
        torch.cuda.empty_cache()

    model_name = "model"
    optimizer_name = "optimizer"
    model_parameter[model_name] = net.state_dict()
    model_parameter[optimizer_name] = optimizer.state_dict()

    torch.save(model_parameter, model_file)

    torch.save(model_parameter, model_file)