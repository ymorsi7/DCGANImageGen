import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

def trainDiscriminator(D, G, optimD, data, batch_size, device):

    '''
    Trains the Discriminator
    Inputs: Discriminator, Generator, Discriminator's optimizer, trainLoader, 128 batch size, device
    Output: loss
    '''
    criterion = nn.BCELoss()
    D.zero_grad()
    realOutput = D(data.to(device))
    realLoss = criterion(realOutput, torch.ones_like(realOutput).to(device))
    realLoss.backward()

    noise = torch.randn(batch_size, 100, 1, 1).to(device)
    fakeOutput = D(G(noise).detach())
    fakeLoss = criterion(fakeOutput,  torch.zeros_like(fakeOutput, dtype = torch.float).to(device))

    fakeLoss.backward()
    optimD.step()
    loss = fakeLoss + realLoss
    return loss

def trainGenerator(D, G, optimG, data, batch_size, device):
    '''
    Trains the Generator
    Inputs: Discriminator, Generator, Generator's optimizer, trainLoader, 128 batch size, device
    Output: loss
    '''
    criterion = nn.BCELoss()
    G.zero_grad()
    noise = torch.randn(batch_size, 100, 1, 1).to(device)
    output = D(G(noise))
    loss = criterion(output, torch.ones_like(output))
    loss.backward()

    optimG.step()
    return loss

def trainProcess(epoch_num, train_loader, D, G, optimD, optimG, dLoss, gLoss, lr, kernSize, padVal, device):
    '''
    Trains the models on epoch_num epochs, saving models and printing results
    Inputs: epoch_num, train loader, discriminator, generator, both optimizers, 
    both loss, learning rate, kernel size, padding, device
    Outputs: loss over iterations
    '''
    for epoch in range(epoch_num):
        for i, (images, _) in enumerate(train_loader):
            dError  = trainDiscriminator(D, G, optimD, images, 128, device)
            gError = trainGenerator(D, G, optimG, images, 128, device)
        dLoss.append(dError.cpu().tolist())
        gLoss.append(gError.cpu().tolist())

    torch.save(G, f'G_{epoch_num}_{lr}_{kernSize}-{padVal}.pt')
    torch.save(D, f'D_{epoch_num}_{lr}_{kernSize}-{padVal}.pt')

    print(f'After {epoch_num} epochs with lr = {lr}, kernel size = {kernSize},
          padding = {padVal}: \tDiscriminator Loss: {dError}\t Generator Loss: {gError}\n')

    return dLoss, gLoss

def trainResults(epoch_num, G, dLoss, gLoss, device):  
    '''
    Plots the results from the training process
    Inputs: epoch_num, generator, loss over iterations, device
    '''
    fake_images = G(torch.randn(64, 100, 1, 1).to(device)).detach().cpu()
    fake_images = vutils.make_grid(fake_images, padding = True)
    plt.figure()
    plt.plot(dLoss, label = "Discriminator")
    plt.plot(gLoss, label = "Generator")
    plt.xlabel("Number of Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Discriminator and Generator Loss Over Epochs")

    plt.figure(figsize = (8, 8))
    plt.axis("off")
    plt.title(f"Fake Images After {epoch_num} Epochs")
    plt.imshow(np.transpose(fake_images,(1,2,0)))
    plt.show()