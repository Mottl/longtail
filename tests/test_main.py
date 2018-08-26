# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import longtail

def test_longtail():
    np.random.seed(42)
    x = np.random.laplace(size=100000)
    scaler = longtail.GaussianScaler()
    x_ = scaler.fit_transform(x)
    mean = np.mean(x_) 
    std = np.std(x_)
    assert mean < 0.001 and mean > -0.001
    assert std < 1.001 and std > 0.999
