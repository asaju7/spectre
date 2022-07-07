#Distributed under the MIT License.
#See LICENSE.txt for details.

import numpy as np

#Functions for testing MomentumDensity.cpp


def momentum_density(pi, phi):
    return np.linalg.norm(pi) * np.linalg.norm(phi)


#End functions for testing MomentumDensity.cpp