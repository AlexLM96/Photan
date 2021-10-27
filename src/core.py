# -*- coding: utf-8 -*-
"""
Introduction of "cont_var" and "photometry" classes.

The cont_var class aims to store any type of continuous data. Right now it does not have much
functionality other than being the superclass of the photometry subclass. In future releases it
will include other subleclasses such as speed, local field potential (LFP) classes, and other types
of continuous data. As a superclass, it will contain methods common to all types of continuous variables
such as peri-event histograms, binning, plotting, etc, which will be inherited by its subclasses.

The photometry class aims to store fiber photometry recordings. It contains information about the
recording, such as sampling rate, timestamps and fluorescence signal. It contains methods for fast
preprocessing of the signal and plotting functionality.
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import seaborn as sns


class cont_var:
    """
    A class used to represent continuous variables.
    
    Attributes
    ----------
    signal: array-like
        The values of the continuous variables
    sr: int
        The sampling rate of the recording
    timestamps: array-like
        The timestamps of the recording
    name: str
        The name of the variable (defaults to None)
    annotations: dict
        Dictionary with annotations about the recording
        
    Methods
    -------
    plot()
        Plots the continuous variable using self.timestamps as the x-axis and self.signal as the y-axis
    
    """
    def __init__(self, signal, sampling_rate, timestamps, name = None, annotations = None):
        """
        Parameters
        ----------
        signal: array-like
            The values of the continuous variables
        sr: int
            The sampling rate of the recording
        timestamps: array-like
            The timestamps of the recording
        name: str
            The name of the variable (defaults to None)
        annotations: dict
            Dictionary with annotations about the recording
            
        """
        self.sr = sampling_rate
        self.signal = signal
        self.timestamps = timestamps
        self.name = name
        self.annotations = annotations

    def plot(self):
        """
        Plots the continuous variable using self.timestamps as the x-axis and self.signal as the y-axis
        """
        fig, ax = plt.subplots(figsize = (6,1))
        ax.plot(self.timestamps, self.signal)
        plt.show()

class photometry(cont_var):
    """
    A subclass of continuous variables that represents photometry recordings.
    
    photometry objects contain the same attributes as the cont_var class
        
    Methods
    -------
    butter_highpass(self, low)
        High-pass butterworth digital filter design
        
    butter_lowpass(self, ihigh)
        Low-pass butterworth digital filter design
        
    debleach(self, low)
        Applies butter_highpass forward and backward to self.signal
        
    lp_filter(self, high)
        Applies butter_lowpass forward and backward to self.signal
        
    zscore(self)
        Z-scores data
    """
    def __init__(self, signal, sampling_rate, timestamps, name = None, annotations = None):
        """
        Parameters
        ----------
        signal: array-like
            The values of the continuous variables
        sr: int
            The sampling rate of the recording
        timestamps: array-like
            The timestamps of the recording
        name: str
            The name of the variable (defaults to None)
        annotations: dict
            Dictionary with annotations about the recording
            
        """
        super().__init__(signal, sampling_rate, timestamps, name, annotations)
    
    def debleach(self, low):
        """
        High-pass filter of photometry signal
        
        Designs high-pass butterworth filter using scipy.signal.butter and 
        applies it to self.signal forward and backward using scipy.signal.filtfilt
        
        
        Parameters
        ----------
        low: float
            Critical low frequency to be filtered
            
        Returns
        ---------
        Photometry object with the debleached signal
        """
        b,a = butter(3, low, btype='high', fs = self.sr)
        y = filtfilt(b, a, self.signal, padtype = "even")
        self.signal = y
        return self
    
    def lp_filter(self, high):
        """
        Low-pass filter of photometry signal
        
        Designs low-pass butterworth filter using scipy.signal.butter and 
        applies it to self.signal forward and backward using scipy.signal.filtfilt
        
        
        Parameters
        ----------
        high: float
            Critical high frequency to be filtered
            
        Returns
        ---------
        Photometry object with the high-pass filtered signal
        """
        b,a = butter(3, high, btype = 'low', fs = self.sr)
        y = filtfilt(b, a, self.signal, padtype = 'even')
        self.signal = y
        return self
    
    def zscore(self):
        """
        Z-scores the photometry signal.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        Z-scored signal
        
        """
        avg = np.mean(self.signal)
        std = np.std(self.signal)
        z_score = (self.signal - avg) / std
        self.signal = z_score
        return self
