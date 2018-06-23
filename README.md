# longtail
Transforms RV from the given empirical distribution to the standard normal distribution

## Usage:
```python
import numpy as np
import longtail

y = np.random.laplace(size=100000)
longtail.plot(y)
```
```
Estimating distributions parameters...
norm (-0.002947709369093035, 1.4120986061330212)
laplace (-0.0015149668735257367, 0.9962681216397508)
cauchy (-0.0016636861308046977, 0.6439073171681272)
```

![](examples/hist_laplace.png?raw=true)  
![](examples/pdf_laplace.png?raw=true)  

```python
scaler = longtail.GaussianScaler()
y_ = scaler.fit_transform(y)
longtail.plot(y_)
```
```
Estimating distributions parameters...
norm (-0.0003123820476865189, 0.9972298229824003)
laplace (-0.006536489035160511, 0.7924577184080439)
cauchy (-0.0040960015181256, 0.6037980352309185)
```

![](examples/hist_normal.png?raw=true)  
![](examples/pdf_normal.png?raw=true)  


## Requirements:
- numpy
- scipy
- matplotlib
