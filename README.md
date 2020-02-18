# White box adversarial attack on MNIST

We use a simple CNN Model to classify MNIST digits.

You can use pre-trained model or can train it by using:
``
python train.py
``


Then perform an attack with
``
python attack.py --test-size 100 --epsilon 0.1
``
Feel free to play with parameter values.

````shell script
usage: attack.py [-h] [--test-size TEST_SIZE] [--epsilon EPSILON]
                 [--target-value {0,1,2,3,4,5,6,7,8,9}] [--targeted]

optional arguments:
  -h, --help            show this help message and exit
  --test-size TEST_SIZE
                        number of sample evaluated (default: 1000)
  --epsilon EPSILON     epsilon value (default: 0.05)
  --target-value {0,1,2,3,4,5,6,7,8,9}
                        Targeted class (default: 0)
  --targeted            Enable class target on fgsm (default: False)

````





Fast Gradient Sign Method (FGSM):

FGSM is used  to trick and perturb model, to cause model it to make mistakes on its predictions.
It is used to evaluate robustness of a trained model on sightly modified data.
By using the gradient of the cost function, it can increase model error and tries to approximate the loss function.

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cboldsymbol%7BX%7D%5E%7Ba%20d%20v%7D%3D%5Cboldsymbol%7BX%7D&plus;%5Cepsilon%20%5Coperatorname%7Bsign%7D%5Cleft%28%5Cnabla_%7BX%7D%20J%5Cleft%28%5Cboldsymbol%7BX%7D%2C%20y_%7Bt%20r%20u%20e%7D%5Cright%29%5Cright%29" />
</p>

From https://arxiv.org/abs/1412.6572

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cboldsymbol%7BX%7D%5E%7Ba%20d%20v%7D%3D%5Cboldsymbol%7BX%7D-%5Cepsilon%20%5Coperatorname%7Bsign%7D%5Cleft%28%5Cnabla_%7BX%7D%20J%5Cleft%28%5Cboldsymbol%7BX%7D%2C%20y_%7Bt%20a%20r%20g%20e%20t%7D%5Cright%29%5Cright%29"/>
</p>

Iterative fast gradient sign method (IFGSM):

Same with FGSM but applied on N gradient steps with alpha = epsilon / N

<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?%5Clarge%20%5Cboldsymbol%7BX%7D_%7B0%7D%5E%7Ba%20d%20v%7D%3D%5Cboldsymbol%7BX%7D%2C%20%5Cquad%20%5Cboldsymbol%7BX%7D_%7BN&plus;1%7D%5E%7Ba%20d%20v%7D%3D%5Coperatorname%7B%5Cemph%7BClip%7D%7D_%7BX%2C%20%5Cepsilon%7D%5Cleft%5C%7B%5Cboldsymbol%7BX%7D_%7BN%7D%5E%7Ba%20d%20v%7D&plus;%5Calpha%20%5Coperatorname%7Bsign%7D%5Cleft%28%5Cnabla_%7BX%7D%20J%5Cleft%28%5Cboldsymbol%7BX%7D_%7BN%7D%5E%7Ba%20d%20v%7D%2C%20y_%7Bt%20r%20u%20e%7D%5Cright%29%5Cright%29%5Cright%5C%7D"/>
</p>

From https://arxiv.org/abs/1611.01236
