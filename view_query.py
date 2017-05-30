#!/usr/bin/env python

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

import argparse

def ImshowNoax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


def SaveQuery(base_dir, line, result_img):
    tokens = line.split(',')
    if (len(tokens)!=6):
        print('Skipping ' + line)
        return
    for i in range(6):
        im_meta = tokens[i].split(':')
        im_path = im_meta[0].strip()
        im_class = im_meta[1].strip()
        im = imread(os.path.join(base_dir, im_path))
        plt.subplot(2,3,i+1)
        ImshowNoax(im, normalize=False)
        plt.title(im_class)
    plt.savefig(result_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View query results')
    parser.add_argument('--queries', type=str, default='',
            help='file with query results')
    parser.add_argument('--images', type=str, default='',
            help='base directory for images')
    parser.add_argument('--results', type=str, default='',
            help='directory to save results to')

    args = parser.parse_args()
    queries = args.queries
    base_dir = args.images
    results_dir = args.results

    lines = open(queries).read()
    qnum = 1
    for line in lines.split('\n'):
        SaveQuery(base_dir, line, os.path.join(results_dir, 'query_%03d.png' % qnum))
        qnum += 1
