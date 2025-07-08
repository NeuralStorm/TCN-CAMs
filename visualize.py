import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from pathlib import Path

# Gets the CAMs from the test data and organizes it into an array with CAMs by class
def get_cams(net, testloader):
    net.eval()
    with torch.no_grad():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        classes = [0,1,2,3,4,5,6,7,8,9]
        heatmap = {classname: [] for classname in classes}
        count = 0
        for images, labels_temp in testloader:
            #just generate CAMs for the first 100 batches
            count = count + 1
            if count > 100:
                break
            images = images.to(device)
            outputs, cams = net(images, generateCAMs = True)
            _, predicted = torch.max(outputs.data, 1)
            labels_temp  = labels_temp.to(device)
            for batch in range(len(labels_temp)):
                h = cams[batch,:].cpu().detach().numpy()
                h = h.squeeze()
                heat = np.array(h[labels_temp[batch],:])
                heatmap[classes[labels_temp[batch]]].append(heat)
        return heatmap

# Plots the CAMs as heatmaps, saving them to the directory "CAMs"
def plot_cams(heats, num_classes):
    for i in range(num_classes):
        fig, ax = plt.subplots()
        column_labels_gamble = [0] * len(heats[i][0])
        column_labels_gamble[0] = -1
        column_labels_gamble[-1] = 0
        heatmap = sns.heatmap(np.reshape(np.array(heats[i][0])[:1170], (30,39)), cmap=sns.color_palette("Spectral", as_cmap=True), ax=ax, cbar=False)
        figure = heatmap.get_figure()  
        plt.yticks(rotation=0)
        images_dir = 'CAMs'
        
        Path(images_dir).mkdir(parents=True, exist_ok=True)
        plt.ylabel("Trial")
        buf = io.BytesIO()
        figure.savefig(f"{images_dir}/" + str(i)+ ".eps", format = "eps")
        buf.seek(0)
        plt.clf()
        plt.close()