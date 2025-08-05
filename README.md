# 🔢 QMNIST Digit Classification using Multi-Layer Perceptron (PyTorch)

This project explores digit classification on the **QMNIST dataset** using a Multi-Layer Perceptron (MLP). It includes robust experimentation with model architecture, loss functions, activation layers, optimizers, and dropout to optimize accuracy.

---


## 🧠 Dataset: QMNIST

* **QMNIST** is an extended version of the classic MNIST handwritten digits dataset.
* 28x28 grayscale images of digits (0–9)
* Comes with training and test sets pre-split via `torchvision.datasets.QMNIST`

---

## 🧪 Objective

> Build, train, and optimize a multi-layer perceptron (MLP) to classify digits in the QMNIST dataset. Analyze the impact of different model hyperparameters.

---

## 🔧 Model Architecture

### 🔹 Baseline MLP

```
Input: 784 (28x28 flattened)
→ Dense(128) + ReLU
→ Dense(64) + ReLU
→ Dense(10) (logits)
```

* **Loss**: CrossEntropyLoss
* **Optimizer**: Adam
* **Batch size**: 64
* **Epochs**: 5

### 🔹 Modified MLP (increased capacity)

```
Input: 784
→ Dense(256) + ReLU
→ Dense(64) + ReLU
→ Dense(10)
```

---

## 📊 Performance

| Model Version                                   | Test Accuracy |
| ----------------------------------------------- | ------------- |
| Baseline MLP                                    | 96.43%        |
| With 256 hidden units                           | 97.24% ✅      |
| Best Hyperparameter Run (SGD, ReLU, No Dropout) | **96.70%**    |
| Worst Hyperparameter Run (SGD, Sigmoid)         | 92.77%        |

---

## 🔬 Hyperparameter Experiments

| Config                                            | Accuracy                   |
| ------------------------------------------------- | -------------------------- |
| **Optimizer**: SGD                                | ✅ Best Performance         |
| **Loss**: CrossEntropy > NLLLoss                  | ✅ Better with raw logits   |
| **Activation**: ReLU > LeakyReLU > Tanh > Sigmoid | ✅ ReLU consistently better |
| **Dropout**: 0.0 (no dropout)                     | ✅ Highest performance      |

> 🧠 ReLU activation with SGD and no dropout gave the best test accuracy (96.70%).

---

## 📉 Training Visualization

* Training loss printed every 100 batches
* Accuracy measured on train/test
* Plots included for accuracy and predictions

---

## 🧪 Sample Predictions

* Predicted vs actual visualizations
* Matplotlib-based plot to display digit and its predicted label

---

## 📌 How to Run
pip install torch torchvision matplotlib
Run all cells in `qmnist_classification.ipynb`

---

## 📘 Key Learnings

* **Neural capacity matters**: Increasing neurons improves performance
* **SGD outperformed Adam**: With right tuning, classic optimizers still shine
* **Dropout isn’t always needed**: Especially for small, clean datasets like MNIST/QMNIST
* **ReLU is generally best**: Simple and effective activation

---

## 🏁 Conclusion

This project showcases how architectural changes and careful hyperparameter tuning can significantly improve digit classification performance using a simple MLP on QMNIST.

---
