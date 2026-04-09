# Generative Models for FashionMNIST Image Synthesis

## Overview
This project investigates how generative model design choices affect image synthesis quality on the FashionMNIST dataset. Three model iterations were implemented and evaluated: a Variational Autoencoder (VAE), a Deep Convolutional GAN (DCGAN), and a Wasserstein GAN with Gradient Penalty (WGAN-GP).

## Objective
The goal of this project was to compare different generative deep learning architectures for image synthesis and analyse how design changes affect image quality, training behaviour, and quantitative performance.

## Dataset
The models were trained and evaluated on the FashionMNIST dataset. The data was split into:
- Training set: 54,000 images
- Validation set: 6,000 images
- Test set: 10,000 images

All images were converted to tensors and normalised to the range [-1, 1].

## Tools and Libraries
- Python
- PyTorch
- torchvision
- torchmetrics
- NumPy
- Matplotlib

## Methodology

### Data preparation
A consistent preprocessing pipeline was applied across all experiments to ensure fair comparison between model iterations. This included tensor conversion, image normalisation, and a fixed train-validation-test split.

### Iteration 1: Variational Autoencoder (VAE)
The first model was a convolutional VAE used as a stable baseline for generative modelling. The VAE learned latent representations through a reconstruction objective combined with KL-divergence regularisation.

### Iteration 2: Deep Convolutional GAN (DCGAN)
The second model replaced the VAE with an adversarial framework using a generator and discriminator. This iteration was designed to test whether adversarial learning could improve image sharpness and visual realism over the baseline.

### Iteration 3: Wasserstein GAN with Gradient Penalty (WGAN-GP)
The third model introduced a critic-based training objective with gradient penalty to improve training stability. This iteration aimed to address instability in standard GAN training while maintaining strong image generation quality.

## Evaluation
Model performance was assessed using:
- qualitative inspection of generated samples
- training and validation behaviour
- MiFID (Memorization-Informed Fréchet Inception Distance)

## Results

| Model | Key Result |
|---|---|
| VAE | Final training loss: 62.5568 |
| VAE | Final validation loss: 63.3653 |
| VAE | MiFID: 1111.5130 |
| DCGAN | Final discriminator loss: 1.0040 |
| DCGAN | Final generator loss: 1.3195 |
| DCGAN | MiFID: 98.9251 |
| WGAN-GP | Final critic loss: -1.8904 |
| WGAN-GP | Final generator loss: 12.1234 |
| WGAN-GP | MiFID: 894.5772 |

## Key Findings
- The VAE provided a stable and interpretable baseline but produced blurrier outputs.
- The DCGAN generated sharper and more realistic images than the VAE.
- The WGAN-GP improved over the VAE baseline but did not outperform DCGAN in this implementation.
- DCGAN delivered the strongest overall balance between visual quality and quantitative performance.

## Repository Contents
- `README.md` — project summary
- `visualcomputing_CS827-2.ipynb` — full notebook with preprocessing, training, evaluation, and comparison
- `generative_models_fashionmnist_report.pdf` — exported project report

## Takeaways
This project demonstrates practical skills in:
- generative deep learning
- iterative model development
- PyTorch model implementation
- adversarial training
- quantitative evaluation with MiFID
- experimental comparison and model analysis
