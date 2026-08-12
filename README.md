# Neural Network From Scratch vs. Keras — IRIS & MNIST

This project builds a neural network completely from scratch in NumPy (forward pass, backprop, gradient descent, all written by hand — no PyTorch or TensorFlow doing the math), then rebuilds the same thing in Keras to compare the two. The output layer uses sigmoid everywhere, as required.

## What's here

`Neural_Network_Project.ipynb` — from-scratch model, Keras model, both datasets, and the comparison, all in one notebook.

## Design choices

**Sigmoid on every layer, not just the output.** Using it in the hidden layers too kept the backward pass consistent (`a * (1-a)` at every layer), which made it much harder to mess up the manual backprop. The trade-off is vanishing gradients on deeper nets — more on that below.

**MSE instead of cross-entropy.** Cross-entropy usually pairs with softmax, but we needed sigmoid on the output. MSE works cleanly with sigmoid regardless of how many output neurons there are, which is also why binary and multi-class ended up using the exact same code.

**One `NeuralNetwork` class for everything.** It just takes a list of layer sizes (`[4, 8, 3]` for IRIS, `[784, 128, 64, 10]` for MNIST). Binary classification isn't a special case — it's just what happens when the last number is 1 instead of something bigger.

**Weight init:** scaled by `sqrt(1/fan_in)` instead of a flat constant, since starting too big saturates sigmoid neurons immediately and nothing learns.

## Binary vs. multi-class

Rather than just explaining the difference, I actually built a binary example: IRIS, versicolor vs. virginica (skipped setosa since it's trivially separable and wouldn't prove much).

- **Binary:** 1 output neuron, sigmoid, threshold at 0.5, labels as plain 0/1.
- **Multi-class** (full IRIS, MNIST): one neuron per class, `argmax` picks the prediction, labels are one-hot.

The backward pass code is identical either way — only the label shape and how the prediction gets read out change.

## Datasets

**IRIS** — small, mostly a sanity check that the math is right.

**MNIST** — this is where pure NumPy actually gets slow, so the from-scratch model trains on 10k images instead of the full 60k. Keras trains on all 60k since it can afford to. That gap is part of the comparison, not hidden.

## Training

Same architecture, same sigmoid setup, same MSE loss for both versions of each dataset — kept identical on purpose so the comparison is about the library itself, not about one version using better tricks. What's allowed to differ: optimizer (plain SGD vs. Adam), data volume for MNIST, and speed.

## Comparison

The notebook includes an accuracy table for both versions on all three problems, training time, and overlaid loss curves.

- **Performance:** IRIS is close for both — within a few points, mostly noise. MNIST shows a real gap, but it's more about data volume and optimizer than the underlying math being wrong.
- **Training behavior:** the scratch loss curve is bumpier since each update sees less data with no momentum smoothing it out. Adam in Keras converges smoother and faster.
- **Limitations from scratch:** slow (no vectorized/GPU backend), plain SGD only, vanishing gradients from all-sigmoid layers on deeper nets, no dropout/batch norm unless built by hand.
- **Advantages of Keras:** handles optimizer/batching/vectorization automatically, architecture changes take one line, and it's less likely to have a silent bug — a wrong transpose in a hand-written gradient doesn't crash, it just trains badly without telling you.

## What I actually learned

Writing the backward pass by hand is what made the chain rule click for real. Once I'd written `delta = (output - y) * sigmoid_deriv(output)` and pushed it backward through a weight matrix myself, "vanishing gradients" stopped being a phrase I'd read somewhere and became something I actually watched happen in a loss curve.

## Running it

Open in Google Colab (has TensorFlow pre-installed) and Run All. Locally you'll need `numpy`, `scikit-learn`, `matplotlib`, `pandas`, `tensorflow`. MNIST downloads automatically.
