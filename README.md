# Clustering and Querying Images Using Metric Learning

This project uses a triplet network to map images to embeddings, that can then be used to cluster similar images, or query similar images. The idea is to use metric learning to learn features for images so that images from the same class are close to each other.

A detailed report about the project is in `docs` directory. The project uses `python 2.7` and `pytorch`, and runs with and without a GPU, but using a  GPU is highly recommended. The project borrows some code from [another triplet network in pytorch](https://github.com/andreasveit/triplet-network-pytorch) and is also inspired by [openface](https://github.com/cmusatyalab/openface).

I have also included result logs generated for experiments for the report, in `docs/report-results`. Each directory in `docs/report-results` is a separate experiment. Please see `docs/report-results/README` for a description of files and results.

The code includes a dataloader for CUB-200-2011 birds dataset. You can download the dataset using download_data.sh.

The file `train.py` feeds in triplets to a network, and computes a loss over these triplets. You can use several different networks and loss functions, and pass in different parameters to the code. Type `./train.py --help` for a full list of options.

The file `train_pushed.py` feeds in a batch of images, with some randomly sampled classes per batch, and some randomly sampled images per class. Instead of feeding in triplets, this accumulates loss from all hard triplets within the batch. This gives many more triplets per batch, and the resulting clustering and querying results are much better.

Type `./train_pushed.py --help` for a full list of options.

You can find example ways to run the `train.py` and `train_pushed.py` in `*-command.sh`.
