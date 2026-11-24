# Assignment 4: _Neural Networks in NumPy_

Students implement a fully-connected feed-forward neural network from scratch using NumPy, then train it to classify handwritten digits from the MNIST dataset. The assignment covers the forward pass, the backward pass via manual gradient computation and the chain rule, mini-batch stochastic gradient descent, and the effect of the learning rate on training dynamics.


## Student Assumptions

**Knows:**
- Python at the level required by prior assignments in this course.
- NumPy array operations, broadcasting, and matrix multiplication.
- Calculus at the level of partial derivatives and the chain rule.
- Basic linear algebra: matrix–vector products, transposes.
- That classification models output a probability distribution over classes (covered in A2).

**Does not yet know:**
- How a neural network is structured or trained.
- What backpropagation is or how gradients flow through a network.
- What a loss function is or why cross-entropy is used for classification.
- What the softmax function is.


## Task Index

```
a — task_prompts/a_introduction.md            — Introduce feed-forward networks and the Module interface
b — task_prompts/b_linear.md                  — Implement the Linear layer
c — task_prompts/c_activation.md              — Implement ReLU, LeakyReLU, and GELU
d — task_prompts/d_loss.md                    — Implement CrossEntropy and MSE
e — task_prompts/e_regularization.md          — Implement Dropout; introduce L2 (implemented in the optimizer task)
f — task_prompts/f_models.md                  — Implement Sequential and MLP
g — task_prompts/g_optim.md                   — Implement the SGD and Adam step methods
h — task_prompts/h_data-handling.md           — Implement Dataset, ArrayDataset, and DataLoader
i — task_prompts/i_metrics.md                 — Implement evaluation metrics: accuracy helper and confusion-matrix functions
j — task_prompts/j_training.md                — Implement the training loop and run a learning-rate experiment
k — task_prompts/k_decision-boundaries.md     — Decision boundaries on 2D toy data (observe-only)
l — task_prompts/l_training-diagnostics.md    — Gradient and weight tracking during training (observe-only)
m — task_prompts/m_regression.md              — Regression: MSE + continuous output (observe-only)
n — task_prompts/n_input-scaling.md           — Input scaling; ToArray/Normalize transforms (observe-only)
o — task_prompts/o_hyperparameter-experiments.md — Optimizers and regularization (observe-only)
p — task_prompts/p_weight-visualization.md    — First-layer weight visualization (observe-only)
```


## Provided vs. To-Implement

**Provided**

- `src/datasets/mnist.py` — MNIST loading utilities; the torchvision-style `MNIST` dataset class and its private cache helpers.
- `src/datasets/toy.py` — toy dataset generators (`make_moons`, `make_circles`, `make_xor`, `make_linear`, `make_blobs`, `make_checkerboard`, `make_spiral`, `make_curve`, `true_curve`) and matching `Dataset` subclasses (`Moons`, `Circles`, `XOR`, `Linear`, `Blobs`, `Checkerboard`, `Spiral`, `Curve`).
- `src/transforms/` — `Compose`, `Normalize`, and `ToArray` transform utilities (`Standardize` was renamed to `ToArray`).
- `src/plots/boundaries.py` — decision boundary plot helpers.
- `src/plots/curves.py` — generic multi-run training curves plot helper.
- `src/plots/diagnostics.py` — training-diagnostic trajectory plots (four-condition comparison grid).
- `src/plots/regression.py` — regression fit plot helpers.
- `src/plots/scaling.py` — input-scaling diagnostic grid plot (3-row × 3-col: val accuracy, grad norm, update magnitude).
- `src/plots/training.py` — learning-rate experiment plot (validation accuracy per epoch, one curve per lr).
- `src/plots/weights.py` — first-layer weight grid plot helpers; `plot_weight_blocks` accepts an optional `"title"` key per block to override the default `H=…, acc=…` label.
- `src/training/get_network.py` — build-and-train helper that integrates with the model cache; used by all `src/experiments/` modules.
- `src/training/model_cache.py` — content-addressed `ModelCache` and `ExperimentConfig`; caches trained models by config + code hash.
- `scripts/experiments.py` — CLI dispatcher for all experiment demos (tasks j–p).
- `src/experiments/lr_sweep.py` — `run_lr_experiment`; runs learning rates `0.01`, `0.1`, `1.0` for 20 epochs and saves a validation-accuracy plot.
- `src/data/utils.py` — `Subset` and `random_split` utilities (mirror `torch.utils.data`).
- `solution.py` — instructor-only answer key (not shipped to students).
- `src/evaluation/metric.py` — `Metric` ABC and `MetricCollection`; provided in full.
- `src/evaluation/gradient_norm.py`, `weight_norm.py`, `grad_weight_ratio.py`, `gradient_histogram.py`, `update_magnitude.py` — Training-diagnostic `Metric` subclasses that monitor model parameters and gradients; each takes the model as a constructor argument.

**To implement**

- `src/nn/linear.py` — `Linear` class (task b).
- `src/nn/activations.py` — `ReLU`, `LeakyReLU`, and `GELU` classes (task c).
- `src/nn/loss.py` — `CrossEntropy` and `MSE` classes (task d).
- `src/nn/dropout.py` — `Dropout` class (task e).
- `src/nn/sequential.py` — `Sequential` class (task f).
- `src/nn/mlp.py` — `MLP` class (task f). `MLP` now accepts a string activation name (`"relu"`, `"leaky_relu"`, `"gelu"`, `"identity"`) and constructs a fresh activation instance per hidden layer.
- `src/evaluation/accuracy.py` — `accuracy` helper (task i); `Accuracy` class is provided.
- `src/evaluation/confusion_matrix.py` — `ConfusionMatrix.update`, `_per_class_precision`, `_per_class_recall`, and `f1` (task i); `__init__`/`reset`/`compute` and the `precision`/`recall`/`accuracy(cm)` wrappers are provided.
- `src/optim/sgd.py` — `SGD.step` (task g); constructor and `zero_grad` inherited from `Optimizer`.
- `src/optim/adam.py` — `Adam.step` (task g); constructor and `zero_grad` inherited from `Optimizer`.
- `src/data/dataset.py` — `Dataset` and `ArrayDataset` classes (task h).
- `src/data/dataloader.py` — `DataLoader` class (task h).
- `src/training/train_ffnn.py` — `run_epoch` and `train_ffnn` (task j).

The `Module` base class in `src/nn/module.py` is provided. Students inherit from it but do not modify it.


## Code and Repo Requirements

- NumPy is allowed; no autograd libraries (PyTorch, JAX, TensorFlow, etc.).
- Full type annotations required; `mypy` is enforced.
- Line length <= 79 characters, PEP 8 throughout (source code files only; `.md` files have no line-length limit).
- All random operations must accept a seed so tests are deterministic.


## Writing Requirements

Assignment-specific notes:
- Backpropagation and the chain rule must be explained from scratch; do not assume the student has seen them before.
- The cross-entropy section must introduce one-hot encoding and softmax before presenting the loss formula.
- Gradient derivations should show both the single-sample and batched (summation) forms so the student can connect the math to the NumPy implementation.
- Prior assignments introduced classification and accuracy; those terms do not need re-explanation.


## Task Prompt Overview

`a_introduction.md`
- Introduce fully-connected feed-forward networks at a high level, using functional-composition notation.
- Introduce the `Module` base class (provided) so students understand `__call__`, `forward`, `backward`, `parameters`, and `step` as a uniform interface modeled after PyTorch's `nn.Module`. Nothing to implement.

`b_linear.md`
- Explain the role of weights, biases, and He initialization.
- Student implements `Linear.__init__`, `Linear.forward`, `Linear.backward`, and `Linear.parameters` in `src/nn/linear.py`.

`c_activation.md`
- Explain why non-linear activation functions are necessary.
- Introduce ReLU, Leaky ReLU, and GELU (tanh approximation) with their derivatives; student implements `forward` and `backward` for `ReLU`, `LeakyReLU`, and `GELU` in `src/nn/activations.py`. GELU stays in pure NumPy via the tanh approximation (no `scipy.special.erf`).

`d_loss.md`
- Introduce cross-entropy loss, one-hot encoding, and softmax (including the numerical-stability trick); derive the combined softmax + cross-entropy gradient; student implements `CrossEntropy.forward` and `CrossEntropy.backward`. The loss takes integer class indices, matching PyTorch's `CrossEntropyLoss`.
- Introduce MSE for regression and contrast it with cross-entropy; student implements `MSE.forward` and `MSE.backward` in `src/nn/loss.py`.

`e_regularization.md`
- Introduce overfitting and regularization; student implements `Dropout` (train/eval behaviour and inverted-dropout scaling) in `src/nn/dropout.py`.
- Introduce L2 regularization as background only; it is implemented as the optimizer's weight-decay term in the next task. (L1 is not covered.)

`f_models.md`
- Explain how layers compose into a forward pass and how gradients propagate in reverse during the backward pass.
- Student implements `Sequential.__init__`, `forward`, `backward`, and `parameters` in `src/nn/sequential.py`, and the convenience `MLP` wrapper (with optional `dropout` parameter) in `src/nn/mlp.py`. `step` is inherited from `Module`.

`g_optim.md`
- Explain the optimizer object and its in-place `(param, grad)` reference contract; introduce vanilla SGD, momentum, Nesterov momentum, and weight decay (L2), then briefly introduce Adam (adaptive moments + bias correction) without heavy theory.
- Student implements `SGD.step` in `src/optim/sgd.py` and `Adam.step` in `src/optim/adam.py`. Both classes inherit from `Optimizer` (`src/optim/optimizer.py`), which provides the shared `zero_grad`; constructors are provided in each subclass.

`h_data-handling.md`
- Introduce the dataset abstraction and Python's `__len__` / `__getitem__` protocol.
- Explain Python abstract base classes (`ABC`, `@abstractmethod`) and why they enforce interface contracts.
- Explain how `DataLoader` wraps a `Dataset` to handle shuffling and batching.
- Student implements `Dataset` and `ArrayDataset` in `src/data/dataset.py`, and `DataLoader` in `src/data/dataloader.py`.

`i_metrics.md`
- Introduce the streaming `Metric` framework (`Metric` ABC, `MetricCollection`) as provided infrastructure; explain the `update`/`compute`/`reset` pattern.
- Student implements the `accuracy` helper in `src/evaluation/accuracy.py`; the `Accuracy` class is provided and shown as an example of the framework.
- Student implements `ConfusionMatrix.update`, `_per_class_precision`, `_per_class_recall`, and `f1` in `src/evaluation/confusion_matrix.py`; `__init__`/`reset`/`compute` and the `precision`/`recall`/`accuracy(cm)` wrappers are provided.

`j_training.md`
- Explain mini-batch SGD: epoch, batch, shuffle-without-replacement, parameter update.
- Student implements `run_epoch` and `train_ffnn` in `src/training/train_ffnn.py`. `run_lr_experiment` in `src/experiments/lr_sweep.py` is provided. The experiment runs learning rates `0.01`, `0.1`, and `1.0` for 20 epochs and saves a validation-accuracy plot; CLI entry point is `scripts/experiments.py lr_sweep` (`make demo-task-j`).
- The default architecture is `784 -> 128 -> 10` with one ReLU.
