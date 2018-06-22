# longtail
**Longtail** plots different best-fit probability density functions (PDFs) for the given data.

## Usage:
```python
import numpy as np
import longtail

y = np.random.randn(100000)
longtail.plot(y)
```
```
Estimating distributions parameters...
norm (0.0035079815092311367, 0.9992993746985821)
laplace (0.006570061006097082, 0.7971030961438007)
cauchy (0.007410253342752503, 0.6114054918233304)
```

![](examples/hist.png?raw=true)  
![](examples/pdf1.png?raw=true)  
![](examples/pdf2.png?raw=true)  

## Requirements:
- numpy
- scipy
- matplotlib
