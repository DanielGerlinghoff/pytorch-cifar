# Train MNIST, CIFAR-10, and CIFAR-100 with PyTorch for E3NE

Training of various neural network models for deployment on FPGA, using the E3NE framework. Un-comment the respective datasets and supported models in main.py. Then run the training script, as shown below. The trained model is stored to the *checkpoint* directory.

```
python3 -uB main.py
```

Supported models:
* LeNet
* CNN by Fang et al.
* VGG-11/13/16/19 (with and without batch normalization)
* AlexNet (with and without batch normalization)

Supported datasets:
* MNIST (28x28 or 32x32)
* CIFAR-10/100
