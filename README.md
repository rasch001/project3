# Repository README

This repository contains three main components, each implemented in separate files:

1. **Neural Network in CUDA/PTX:**  
   Implements a neural network using low-level CUDA and PTX code for forward propagation (and attempted backpropagation).

2. **RL Agent for PTX Code Generation:**  
   Uses a reinforcement learning (RL) agent to generate syntactically and semantically correct PTX code from a SASS-like grammar. The agent is trained with PPO on a grammar-based environment.

3. **Simulation of Attention as an Interacting Particle System:**  
   Simulates an interacting particle system to model attention mechanisms, where particles update their positions based on Gaussian interaction kernels derived from a mathematical theory of attention.

Additionally, note that the files **agent_output.ptx**, **big_mats.txt**, **generated_agent.ptx**, and **matrices.txt** all belong to `project_3_file_1.ipynb`. It is suggested that you review the contents of **file 2** first, then **file 1** and **file 3** in any order.

---

## 1. Neural Network, CUDA, and PTX – Forward Propagation

The neural network implemented in CUDA/PTX has the following architecture:
- **Input Layer:**  
  Accepts a flattened MNIST image of dimension $784$ (i.e., $28 \times 28$ pixels).
- **Hidden Layer:**  
  Contains $128$ neurons using the ReLU activation function.
- **Output Layer:**  
  Consists of $10$ neurons, one per class, with a linear activation.

**CUDA and PTX Overview:**  
CUDA is NVIDIA’s platform for parallel computing on GPUs. CUDA code is compiled into PTX (Parallel Thread Execution), a low-level, assembly-like language. PTX code is then further compiled by the GPU driver into device-specific machine code. This low-level control allows for fine-grained optimization and management of parallel computations.

**Forward Propagation in PTX:**  
Each thread in the kernel computes the forward pass for one sample:
- **Layer 1:**  
  $$ a^{(1)} = \max\Big(0, \sum_{j=1}^{784} W^{(1)}_{ij} \, x_j + b^{(1)}_i \Big) $$
- **Layer 2 (Output):**  
  $$ a^{(2)} = \sum_{j=1}^{128} W^{(2)}_{ij} \, a^{(1)}_j + b^{(2)}_i $$
  
These operations are implemented using explicit loops, pointer arithmetic, and type conversions to ensure correct computation on the GPU.

---

## 2. Backpropagation Computation in PTX

The backpropagation algorithm for our network computes gradients using a squared-error loss on one-hot targets:
1. **Output Layer Backpropagation:**  
   The error at the output is computed as
   $$ \delta^{(2)} = a^{(2)} - y, $$
   where $y$ is the one-hot encoded target. The gradients for the output biases are $\delta^{(2)}$, and the gradients for the output weights are computed via
   $$ \nabla W^{(2)} = \delta^{(2)} \otimes a^{(1)}. $$

2. **Hidden Layer Backpropagation:**  
   The hidden layer error is computed as
   $$ \delta^{(1)} = \Big((W^{(2)})^T \delta^{(2)}\Big) \odot \mathbf{1}_{\{a^{(1)} > 0\}}, $$
   where $\mathbf{1}_{\{a^{(1)} > 0\}}$ is the indicator function of the ReLU derivative. The gradients for the hidden biases and weights are then
   $$ \nabla W^{(1)} = \delta^{(1)} \otimes x. $$

**PTX Implementation Notes:**  
In the PTX (or CUDA C) code, these steps are implemented using loops and temporary storage. However, due to PTX’s limitations (such as difficulties with register array indexing and strict type matching), the backpropagation code encountered issues and did not work correctly without significant debugging.

---

## 3. Training Setup Implementation in CUDA

The training procedure is implemented with two main CUDA kernels:

1. **trainStep Kernel:**  
   - **Inputs:** Mini-batch of images and one-hot encoded labels.
   - **Forward Pass:**  
     - Computes hidden layer activations using  
       $$ a^{(1)} = \max\Big(0, W^{(1)} x + b^{(1)}\Big). $$
     - Computes the output as  
       $$ a^{(2)} = W^{(2)} a^{(1)} + b^{(2)}. $$
   - **Backward Pass:**  
     - Computes the output error  
       $$ \delta^{(2)} = a^{(2)} - y, $$
       and backpropagates this error to compute the hidden layer error  
       $$ \delta^{(1)} = \Big((W^{(2)})^T \delta^{(2)}\Big) \odot \mathbf{1}_{\{a^{(1)}>0\}}. $$
     - Calculates per-sample gradients and stores them in gradient buffers.
     
2. **updateWeights Kernel:**  
   - **Function:**  
     Aggregates per-sample gradients over the mini-batch, averages them, and updates each parameter using  
     $$ W \leftarrow W - \gamma \cdot \frac{1}{N}\sum_{i=1}^N \nabla W_i, $$
     where $\gamma$ is the learning rate and $N$ is the batch size.

**Training Loop:**  
- Data is copied from the host to the GPU.
- The `trainStep` kernel computes forward and backward passes.
- The `updateWeights` kernel updates the network parameters.
- Periodically, parameters are copied back to the host for evaluation via a NumPy forward pass.
- This loop continues until the test accuracy exceeds 60% or a preset maximum number of batches is processed.

---

## 4. Comparison with PyTorch

The PyTorch implementation was set up to match the architecture and training procedure of our custom CUDA solution. In this setup:

- **Architecture:**  
  A two-layer network is used with one hidden layer of $128$ neurons (with ReLU activation) and an output layer of $10$ neurons. This exactly mirrors the network used in the PyCUDA implementation.

- **Loss Function and Update Rule:**  
  Both implementations use the Mean Squared Error (MSE) loss on one-hot encoded targets. The PyTorch model computes gradients on a mini-batch and updates parameters with a full batch gradient update, just like the CUDA kernels that average per-sample gradients over the mini-batch.

- **Training Details:**  
  Both models are trained on the MNIST dataset with a batch size of $64$. The training loop in PyTorch updates the network after each mini-batch and evaluates test accuracy every 100 batches, continuing until test accuracy exceeds 60% or a maximum number of batches is reached.

- **Results:**  
  The observed training progress was as follows:
  - **PyCUDA Results:**
    ```
    PyCUDA: Batch 100, Test Accuracy: 13.01%
    PyCUDA: Batch 200, Test Accuracy: 14.63%
    PyCUDA: Batch 300, Test Accuracy: 17.43%
    PyCUDA: Batch 400, Test Accuracy: 22.13%
    PyCUDA: Batch 500, Test Accuracy: 27.75%
    PyCUDA: Batch 600, Test Accuracy: 35.57%
    PyCUDA: Batch 700, Test Accuracy: 42.39%
    PyCUDA: Batch 800, Test Accuracy: 48.74%
    PyCUDA: Batch 900, Test Accuracy: 51.74%
    PyCUDA: Batch 1000, Test Accuracy: 55.54%
    PyCUDA: Batch 1100, Test Accuracy: 58.98%
    PyCUDA: Batch 1200, Test Accuracy: 61.64%
    ```
    Total training time was **33.4456 seconds** over **1200 batches**, reaching a test accuracy of **61.64%**.
    
  - **PyTorch Results:**
    ```
    PyTorch: Batch 100, Test Accuracy: 12.18%
    PyTorch: Batch 200, Test Accuracy: 13.38%
    PyTorch: Batch 300, Test Accuracy: 15.13%
    PyTorch: Batch 400, Test Accuracy: 17.71%
    PyTorch: Batch 500, Test Accuracy: 19.94%
    PyTorch: Batch 600, Test Accuracy: 22.34%
    ...
    PyTorch: Batch 4200, Test Accuracy: 60.33%
    ```
    Total training time was **67.6265 seconds** over **4200 batches**, reaching a test accuracy of **60.33%**.

**Analysis:**  
Both methods were trained on the same amount of data with an identical network architecture and loss function. The PyTorch implementation, while easier to develop and more robust, required more batches and nearly twice the time to reach around 60% accuracy compared to our custom PyCUDA solution. This illustrates that, with careful low-level optimization, custom CUDA code can achieve competitive training performance, albeit at the cost of increased development complexity.

---

**Additional File Notes:**

- The files **agent_output.ptx**, **big_mats.txt**, **generated_agent.ptx**, and **matrices.txt** all belong to `project_3_file_1.ipynb`.
- It is suggested that you read **file 2** (the RL Agent for PTX Code Generation) first, and then review **file 1** (Neural Network in CUDA/PTX) and **file 3** (Simulation of Attention as an Interacting Particle System) in any order.

Happy coding and exploring the intricacies of low-level GPU programming and reinforcement learning!

