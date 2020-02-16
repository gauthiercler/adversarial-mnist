# White box adversarial attack on MNIST

We use a simple CNN Model to classifiy MNIST digits.

You can use pretrained model or can train it by using:
``
python train.py
``


Then perform an attack with
``
python attack.py --test-size 100 --epsilon 0.1
``
Feel free to play with parameter values.

Fast Gradient Sign Method (FGSM) is used  to trick and perturbate model, to cause model it to make mistakes on its predictions.
It is used to evaluate robustness of a trained model on sightly modified data.



Fast Gradient Sign Method:



By using the gradient of the cost function, we can increase model error;


