# Neural Network From Scratch vs. Keras — IRIS & MNIST

This project builds a neural network completely from scratch in NumPy (forward pass, backprop, gradient descent, then rebuilds the same thing in Keras to compare the two. The output layer uses sigmoid everywhere, as required.

## What's here

`Neural_Network_Project.ipynb` — from-scratch model, Keras model, both datasets, and the comparison, all in one notebook.

## Design choices

**Sigmoid on every layer, not just the output.** Using it in the hidden layers too kept the backward pass consistent (`a * (1-a)` at every layer), which made it much harder to mess up the manual backprop.

**One `NeuralNetwork` class for everything.** It just takes a list of layer sizes (`[4, 8, 3]` for IRIS, `[784, 128, 64, 10]` for MNIST). Binary classification isn't a special case — it's just what happens when the last number is 1 instead of something bigger.

**Weight init:** scaled by `sqrt(1/fan_in)` instead of a flat constant, since starting too big saturates sigmoid neurons immediately and nothing learns.

## Binary vs. multi-class

Rather than just explaining the difference, Built a binary example: IRIS, versicolor vs. virginica.

- **Binary:** 1 output neuron, sigmoid, threshold at 0.5, labels as plain 0/1.
- **Multi-class** (full IRIS, MNIST): one neuron per class, `argmax` picks the prediction, labels are one-hot.

The backward pass code is identical either way — only the label shape and how the prediction gets read out change.

## Datasets

**IRIS**,
**MNIST** 

## Training

Same architecture, same sigmoid setup, same MSE loss for both versions of each dataset — kept identical on purpose so the comparison is about the library itself, not about one version using better tricks.

## Comparison

The notebook includes an accuracy table for both versions on all three problems, training time.
