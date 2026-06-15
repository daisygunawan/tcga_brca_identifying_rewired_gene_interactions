"""
Analysis Utility Functions

This module provides statistical analysis functions for co-expression network analysis,
particularly focused on differential correlation testing between tumor and normal samples.

Functions:
    fisher_z_transform_p_value: Calculate p-value for difference between two correlation 
                               coefficients using Fisher's Z-transform method
"""

import pandas as pd
from scipy import stats
import numpy as np

def fisher_z_transform_p_value(r1, n1, r2, n2):
    """
    Calculates the p-value for the difference between two correlation coefficients
    using Fisher's Z-transform method.
    
    This method is used to test whether two correlation coefficients are significantly
    different from each other, commonly applied in differential co-expression analysis.
    
    Parameters:
    -----------
    r1 : float
        Correlation coefficient from the first group (e.g., normal samples)
    n1 : int
        Sample size of the first group
    r2 : float
        Correlation coefficient from the second group (e.g., tumor samples)  
    n2 : int
        Sample size of the second group
        
    Returns:
    --------
    float
        Two-tailed p-value for the difference between the two correlation coefficients
    """
    # Fisher's Z-transform: convert correlation coefficients to approximately normal distribution
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    
    # Calculate standard error of the difference between Z-transformed coefficients
    se_diff = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    
    # Compute Z-score for the difference
    z_diff = (z1 - z2) / se_diff
    
    # Calculate two-tailed p-value from normal distribution
    p_value = 2 * (1 - stats.norm.cdf(abs(z_diff)))
    
    return p_value