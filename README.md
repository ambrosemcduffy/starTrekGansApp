# Star Trek: The Next Generation Faces with wGAN-GP

This project implements a Wasserstein Generative Adversarial Network with Gradient Penalty (wGAN-GP) to generate synthetic faces of characters from the "Star Trek: The Next Generation" series. The project uses PyTorch, a popular deep learning framework, to build and train the model.

## Features

- **wGAN-GP Model:** Utilizes the wGAN-GP architecture for improved training stability and generation quality over traditional GANs.
- **Inception Score and FID Calculations:** Includes functionality to calculate the Frechet Inception Distance (FID) for assessing the quality of generated images.
- **TensorBoard Integration:** Provides TensorBoard logging for tracking the training process and visualizing results.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- NumPy
- matplotlib

## Installation

1. Clone the repository.
2. Install the required dependencies: `pip install -r requirements.txt`.

## Usage

1. Prepare your dataset of Star Trek TNG faces in a folder.
2. Set the `dataDirectory` variable to point to your dataset folder.
3. Run the script to start training: `python wGanGpStarTrek.py`.

## Code Structure

- `wGanModel.py`: Contains the `Generator` and `Discriminator` classes.
    - `Discriminator`: Consists of three convolutional layers with leaky ReLU activation and a fully connected layer. It aims to classify images as real or generated.
    - `Generator`: Comprises five transposed convolutional layers with batch normalization and ReLU activation, ending with a tanh activation layer. It generates synthetic images from the latent space.
- `ganUtils.py`: Provides utility functions like weight initialization and FID calculation.
- `wGanGpStarTrek.py`: Main script for training the model.

## Hyperparameters

- `zDim`: Dimension of the latent space.
- `gHiddenDim` & `dHidden`: Dimensions of hidden layers in the generator and discriminator.
- `lr`: Learning rate.
- `batchSize`: Batch size for training.
- Other parameters like `lambda_gp` and `n_epochs` are set based on the paper and can be tweaked.

## Training

- The script trains the wGAN-GP model on the Star Trek TNG faces dataset.
- During training, loss values and FID scores are logged in TensorBoard.
- The model and TensorBoard logs are saved periodically.

## Outputs

- The model generates synthetic faces of Star Trek TNG characters.
- Generated images can be viewed in TensorBoard or saved to a directory.

## Conclusion

This project provides a fascinating application of wGAN-GP to generate synthetic faces of popular TV show characters. It showcases the potential of GANs in creative content generation and offers a starting point for those interested in exploring advanced GAN architectures.
![Star Trek TNG Faces GAN Demo](https://github.com/ambrosemcduffy/starTrekGansApp/blob/main/gansApp.gif?raw=true)

---
