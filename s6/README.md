# Solution : TSAI_S6

## PART 1 

### Steps to calculate gradients:
- The Formulae section refers to the forward pass. Here, the outputs are updated based on the inputs and weights. At last, the total loss is calculated.
- The Output layer and the Hidden layer blocks refer to the gradient calculation of the network loss w.r.t output and the hidden layer weights. This block provides the formulae for the gradient calculation.
- The Derivatives block contain helper derivatives to compute the final derivative for every weight.
- The Derivatives section (highlighted) in the table calculates the gradient for each weight in the network.
- Based on the gradient, the corresponding weights are updated.
- The total loss is calculated everytime after all the weights are updated. The calculations are performed ~100 times to reduce the loss which can also be noted from the plot.

### Learning rate changes

lr=.1<br>
![image](https://github.com/bijonguha/tsai-bijon/blob/main/images/lr_0.1.png)

lr=.2<br>
![image](https://github.com/bijonguha/tsai-bijon/blob/main/images/l_0.2.png)

lr=.5<br>
![image](https://github.com/bijonguha/tsai-bijon/blob/main/images/lr_0.5.png)

lr=.8<br>
![image](https://github.com/bijonguha/tsai-bijon/blob/main/images/lr_0.8.png)

lr=1<br>
![image](https://github.com/bijonguha/tsai-bijon/blob/main/images/lr_1.png)

lr=2<br>
![image](https://github.com/bijonguha/tsai-bijon/blob/main/images/lr_2.png)

```
Observation : Loss significantly decreases and goes to zero in fewer iterations as we boost the learning rate from 0.1 to 2.
```

## PART 2

### Objective <br>
Achieve 99.4% or more accuracy on mnist dataset with below constraints -
<li> Less than 20k parameters
<li> Less than 20 epochs
<li> Use Batch Normalization
<li> Use Dropout
<li> Fully connected layer, GAP are optional

## Model Summary
Here we have used 18,738 parameters in total and which is aligned with constraint of using less than 20k parameters
![image](./images/Screenshot_model_summary.png)
  
## Result
  
<li> Best Accuracy Test set : 99.43
<li> Epoch : 18
<li> Model Params : 18,738