import torch
import seaborn as sns
from model import GeneratorGAN, DiscriminatorGAN
import matplotlib.pyplot as plt

def testResults(test_loader, num_epochs, lr, kernSize, padVal):
    '''
    Computes test results of classifying real and fake images
    
    '''
    genModel = GeneratorGAN(kernSize = 5, padVal = 2)
    discrimModel = DiscriminatorGAN(kernSize = 5, padVal = 2)

    fake_output_list = []
    real_output_list = []

    genModel.load_state_dict(torch.load(f'G_{num_epochs}_{lr}_{kernSize}-{padVal}.pt'))
    discrimModel.load_state_dict(torch.load(f'D_{num_epochs}_{lr}_{kernSize}-{padVal}.pt'))

    for i, (images, _) in enumerate(test_loader):
        fake_images = genModel(torch.randn(128, 100, 1, 1))
        fake_output = discrimModel(fake_images)
        real_output = discrimModel(images)
        fake_output_list += fake_output.view(-1).tolist()
        real_output_list += real_output.view(-1).tolist()

    fake_output_results = [1 if val >= 0.5 else 0 for val in fake_output_list]
    real_output_results = [1 if val >= 0.5 else 0 for val in real_output_list]
    return fake_output_results, real_output_results

def testPlotting(fake_output_results, real_output_results):
    '''
    Plots confusion matrix using results from testResults
    Inputs: fake_results, real_results from testResluts
    '''
    count = [[0, 0], [0, 0]]
    for i in fake_output_results:
        count[0][i] += 1
    for j in real_output_results:
        count[1][j] += 1
    sns.heatmap(count, annot = True, fmt = 'd', cmap = "Blues", xticklabels = ['Fake', 'Real'], yticklabels = ['Fake', 'Real'])
    plt.title("Confusion Matrix of DCGAN Model After Being Trained on 200 Epochs")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")