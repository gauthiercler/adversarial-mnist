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

Fast Gradient Sign Method (FGSM) is used  to trick and perturb model, to cause model it to make mistakes on its predictions.
It is used to evaluate robustness of a trained model on sightly modified data.



Fast Gradient Sign Method:


By using the gradient of the cost function, we can increase model error

It tries to approximate the loss function


