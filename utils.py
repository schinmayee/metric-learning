import os
from PIL import Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def DefaultImageLoader(path):
    return Image.open(path).convert('RGB')

def Resize(img, size):
    im_size = img.size
    if im_size[0] == im_size[1]:
        new_img = img
    elif im_size[0] > im_size[1]:
        p = (im_size[0] - im_size[1])/2
        new_img = Image.new('RGB', (size, size))
        new_img.paste(img, (p, 0))
    elif im_size[0] < im_size[1]:
        p = (im_size[1] - im_size[0])/2
        new_img = Image.new('RGB', (size, size))
        new_img.paste(img, (0, p))
    new_img = new_img.resize((size, size))
    return new_img

def PlotLosses(file_name, epochs, train_losses, test_losses):
    fig, ax = plt.subplots()
    h1 = plt.scatter(epochs, train_losses, marker='o', color='#d8b365')
    h2 = plt.scatter(epochs, test_losses, marker='^', color='#5ab4ac')
    plt.legend([h1, h2], ['train', 'test'], loc='upper right')
    plt.savefig(file_name)

def PlotAccuracy(file_name, epochs, triplet_accs, classification_accs):
    fig, ax = plt.subplots()
    h1 = plt.scatter(epochs, triplet_accs, marker='o', color='#d8b365')
    h2 = plt.scatter(epochs, classification_accs, marker='^', color='#5ab4ac')
    plt.legend([h1, h2], ['triplet', 'k-means'], loc='upper right')
    plt.savefig(file_name)

def SavePlots(base_dir, epochs, train_losses, test_losses,
              triplet_accs, classification_accs):
    plot1_file = os.path.join(base_dir, 'losses.png')
    plot2_file = os.path.join(base_dir, 'accuracy.png')
    PlotLosses(plot1_file, epochs, train_losses, test_losses)
    PlotAccuracy(plot2_file, epochs, triplet_accs, classification_accs)
