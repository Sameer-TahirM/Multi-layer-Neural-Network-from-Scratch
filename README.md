# Neural Network from Scratch with NumPy

## Description

This project is a foundational implementation of a multi-layer neural network for classifying the Iris dataset. The entire network, including forward propagation, backpropagation, and the loss function, is built from scratch using only **NumPy**. This approach avoids high-level deep learning frameworks like TensorFlow or PyTorch to provide a clear, fundamental understanding of how neural networks operate.

The network uses the **ReLU** activation function for its hidden layers and a **Multi-class SVM (Hinge) Loss** function to handle multi-class classification, making it a robust and educational example of low-level neural network design.

## How It Works

The `NeuralNetwork` class encapsulates the entire model logic:

1.  **Initialization**: The network's weights and biases are initialized with random uniform values based on the specified number of layers and neurons.
2.  **Forward Propagation**: For each training iteration, input data is passed through the network. At each layer, the output is calculated as `A = ReLU(W*X + b)`. The final layer produces raw scores (logits) for each class.
3.  **Loss Calculation**: The model uses a **Multi-class SVM (Hinge) Loss** function. For each example, it computes the loss based on the margin between the score of the correct class and the scores of incorrect classes.
4.  **Backward Propagation**: Gradients of the loss with respect to the weights and biases are calculated using the chain rule, starting from the output layer and working backward.
5.  **Parameter Update**: The weights and biases are updated using standard gradient descent with a defined learning rate.

## Features

- **Built Entirely from Scratch**: No auto-differentiation or high-level libraries used for the core network logic.
- **Pure NumPy Implementation**: All matrix operations, activations, and gradient calculations are done with NumPy.
- **Customizable Architecture**: The number of layers and the number of neurons in each layer can be easily configured.
- **ReLU Activation**: Uses the Rectified Linear Unit (ReLU) activation function in its hidden layers.
- **Multi-class SVM Loss**: Implements the Hinge Loss function, suitable for multi-class classification tasks.
- **Full Training Pipeline**: Includes data loading, preprocessing, a custom train/test split, and a complete training loop.

## Usage

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your_username/your_repo_name.git
    cd your_repo_name
    ```
2.  **Install Dependencies**:
    ```bash
    install numpy and pandas
    ```
3.  **Prepare the Data**:
    - Make sure the `Iris.csv` file is in the same directory as the notebook.

4.  **Run the Notebook**:
    - Open `NN-code.ipynb` in a Jupyter environment.
    - Run the cells from top to bottom to load the data, define the `NeuralNetwork` class, and train the model.

5.  **Train the Model**:
    - The `NeuralNetwork` class is instantiated and trained within the notebook. You can configure the architecture and hyperparameters directly.
    ```python
    # Example of instantiating the network
    # 4 input features, 2 hidden layers with 5 and 2 neurons, and 3 output classes
    nn = NeuralNetwork(
        X_train,
        Y_train,
        hidden_layers=4,
        lst_nodes=[X_train.shape[1], 5, 2, 3],
        lr=0.0001,
        epochs=1000
    )
    ```

    ## Dataset
     Iris dataset
