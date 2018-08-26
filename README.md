# longtail
**Longtail** transforms RV from the given empirical distribution to the standard normal distribution.

![Python 3x](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3x](https://img.shields.io/badge/python-3.x-blue.svg)
[![Build Status](https://travis-ci.org/Mottl/longtail.svg?branch=master)](https://travis-ci.org/Mottl/longtail)

## Usage
```python
import numpy as np
import longtail

x = np.random.laplace(size=100000)
longtail.plot(x)
```
```
Estimating distributions parameters...
norm (0.0018876631529621596, 1.4211949719613757)
laplace (0.0017959270327248976, 1.004473511770026)
cauchy (0.0008371221028761951, 0.6495115549176855)
```

![](examples/hist_laplace.png?raw=true)  
![](examples/pdf_laplace.png?raw=true)  

```python
scaler = longtail.GaussianScaler()
x_ = scaler.fit_transform(x)
longtail.plot(x_)
```
```
Estimating distributions parameters...
norm (5.4850160001935534e-05, 0.9999632908186453)
laplace (5.727594101276392e-05, 0.7978789360688088)
cauchy (-1.422402203800512e-06, 0.6119807936005598)
```

![](examples/hist_normal.png?raw=true)  
![](examples/pdf_normal.png?raw=true)  

```python
plt.plot(scaler.transform_table[:,0], scaler.transform_table[:,1],
    color="dodgerblue", label="Laplace to Gaussian transformation")
plt.title("Transformation function")
plt.xlabel(r"$x$")
plt.ylabel(r"$\hat{x}$", rotation=0)
plt.axis("equal")
plt.grid(True)
plt.show()
```

![](examples/transform_function.png?raw=true)  

## Requirements
- Python 3
- numpy
- scipy
- matplotlib
