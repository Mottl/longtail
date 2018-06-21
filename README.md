# isgaussian
Plot data probability distribution to visualise whether the data can be considered as Normal distributed

## Usage:
```python
import numpy as np
import isgaussian

y = np.random.randn(100000)
isgaussian.plot(y)
```
```
Estimating distributions parameters...
norm (-0.0018456060682745922, 1.0001900711528913)
laplace (-0.0016155583094764148, 0.7982866738374146)
cauchy (-0.002052229009204931, 0.6125983563697108)
```

![](examples/hist.png?raw=true)  
![](examples/pdf1.png?raw=true)  
![](examples/pdf2.png?raw=true)  
