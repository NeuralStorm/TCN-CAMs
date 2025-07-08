import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
import tensorflow as tf
import torchvision.transforms as transforms
import tqdm
from collections import deque
from statistics import mean, stdev
from pathlib import Path

from visualize import get_cams, plot_cams
from model import tcn

def train(epochs = 100, batch_size = 8, learning_rate = .01, start_epoch = 0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)), # Normalizes to [-1, 1]
        transforms.Lambda(lambda x: x.view(x.shape[0], -1)) 
    ])
    trainset = dataset = torchvision.datasets.MNIST(root='.', train=True,
                                download=True, transform = transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = True)

    testset = dataset = torchvision.datasets.MNIST(root='.', train=False,
                                download=True, transform = transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    net = tcn(1, num_classes = 10)
    
    if start_epoch > 0:
        PATH = 'models/epoch ' + str(start_epoch)
        net.load_state_dict(torch.load(f"{PATH}/epoch " + str(start_epoch), weights_only=True, map_location=torch.device('cpu')))

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    for epoch in range(start_epoch, start_epoch + epochs): 
        running_loss = deque()
   
        net.train()
        val_steps = 0
        pbar = tqdm.tqdm(trainloader)
        for inputs, ind_labels in pbar:

            optimizer.zero_grad()
            inputs = torch.FloatTensor(inputs)

            inputs = inputs.to(device)
            outputs, cams = net(inputs)
            ind_labels = ind_labels.to(device)
            loss = criterion(outputs, ind_labels)
            loss.backward()
            optimizer.step()
            val_steps += 1
            running_loss.append(int(loss))
            if len(running_loss) > 1000:
                running_loss.popleft()
            pbar.set_description("Epoch: " + str(epoch) +  ", Loss: "+ str(mean(running_loss)))
        
        PATH = 'models/epoch ' + str(epoch + 1)
        Path(PATH).mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), f"{PATH}/epoch " + str(epoch))

        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        if epoch % 5 == 0:
            correct  = test(net, testloader)
            print("Test accuracy: " + str(correct/(len(trainloader) * batch_size)))
            h = get_cams(net, testloader)
            image = plot_cams(h, 10)

    return net, testloader

#Uses the model to make predictions about the test data, returning the number of correctly classified trials
def test(net, testloader):
    net.eval()
    correct_epoch = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        for data, label in testloader:
            data = data.to(device)
            output, cams = net(data)
            _, predicted = torch.max(output.data, 1)
            labels  = label
            labels  = labels.to(device)
            correct_epoch += (predicted == labels).sum().item()
    return correct_epoch