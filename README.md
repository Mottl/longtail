# isgaussian
Plot data probability distribution to visualise whether the data can be considered as Normal distributed

## Usage:
```python
import numpy as np
import isgaussian

y = np.random.randn(100000)
isgaussian.plot(y)
```

![](examples/hist.png?raw=true)  
![](examples/pdf1.png?raw=true)  
![](examples/pdf2.png?raw=true)  
