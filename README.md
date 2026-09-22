# Simple Linear Model with NumPy

This is a very simple example of a **linear model** using Python and NumPy.

The model learns the relationship between **hours studied (`x`)** and **grade (`y`)**.

## 1. Import NumPy

```python
import numpy as np
```

NumPy is used to create and work with arrays.

## 2. Create the weight

```python
w = 1.0
```

`w` is the **weight** of our model.

The model uses this formula:

```text
y_hat = x × w
```

At the beginning:

```text
w = 1
```

So if:

```text
x = 2
```

the prediction is:

```text
y_hat = 2 × 1 = 2
```

## 3. Forward function

```python
def forward(x):
    return x * w
```

The `forward()` function calculates the model's prediction.

Mathematically:

```text
ŷ = xw
```

Where:

* `x` → input
* `w` → weight
* `ŷ` → predicted value

## 4. Loss function

```python
def loss(x, y):
    y_hat = forward(x)
    return (y_hat - y) ** 2
```

The `loss()` function measures how far the prediction is from the real value.

The formula is:

```text
Loss = (ŷ - y)²
```

Where:

* `ŷ` → predicted value
* `y` → real value
* `Loss` → prediction error

For example:

```text
x = 2
y = 4
w = 1

ŷ = 2 × 1 = 2

Loss = (2 - 4)²
     = (-2)²
     = 4
```

So the model has an error of `4`.

## 5. Training data

```python
x_soat = np.array([1, 2, 3])
y_baho = np.array([2, 4, 6])
```

We have three examples:

| Hours (`x`) | Real grade (`y`) |
| ----------: | ---------------: |
|           1 |                2 |
|           2 |                4 |
|           3 |                6 |

The relationship is:

```text
y = 2x
```

But our model currently has:

```text
w = 1
```

Therefore, the model's predictions are not correct yet.

## 6. `zip()` and the loop

```python
for x, y in zip(x_soat, y_baho):
    loss(x, y)

    print("x: ", x, "y: ", y, "Loss: ", loss(x, y))
```

`zip()` combines the two arrays element by element:

```text
x = 1, y = 2
x = 2, y = 4
x = 3, y = 6
```

For every pair, the program calculates the loss and prints it.

### Output

```text
x:  1 y:  2 Loss:  1.0
x:  2 y:  4 Loss:  4.0
x:  3 y:  6 Loss:  9.0
```

## What is happening?

The model starts with:

```text
w = 1
```

So its predictions are:

```text
x = 1 → ŷ = 1
x = 2 → ŷ = 2
x = 3 → ŷ = 3
```

But the correct answers are:

```text
1 → 2
2 → 4
3 → 6
```

Therefore, the losses are:

```text
(1 - 2)² = 1
(2 - 4)² = 4
(3 - 6)² = 9
```

This code **does not train the model yet**. It only performs a forward pass and calculates the loss.

The next step would be to calculate the **gradient** and update `w` using **Gradient Descent**.
