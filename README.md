Loads NWPU-RESISC45 images and labels from folders, applies resizing + tensor conversion

Samples episodic tasks (5-way 5-shot with 15 queries per class)

Uses a ViT backbone with prompt tokens prepended and tuned

Performs MAML-style inner loop updates only on prompt tokens

Runs meta-training and evaluation loops reporting accuracy and classification metrics

Plots metrics after training

Change dataset_dir to your actual NWPU-RESISC45 dataset folder path

Optionally add normalization transforms if your model requires

Make sure dependencies are installed: pip install torch torchvision timm tqdm scikit-learn matplotlib pillow
