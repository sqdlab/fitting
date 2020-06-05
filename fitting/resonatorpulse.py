import numpy
import scipy.stats
from collections import OrderedDict
from . import FitBase, apply_along_axis

class Resonatorpulse(FitBase):
        PARAMETERS = ['length', 'start', 'amplitude', 'T1']
    
        @staticmethod
        def f(f, length, start, amplitude, T1):
            '''
            amplitude / (1-exp(-length/T1)) * piecewise(f, [ f<start, ( (f>=start)  & (f<(start+length)) ), 
                                                 f>=(start+length)], 
                                             [0,
                                              lambda f: (1-exp( -(f-start) / T1)), 
                                             lambda f: exp(-(f-start-length) / T1) * (1-exp(-length/T1)), 
                                             ] )
            '''
            return amplitude / (1-numpy.exp(-length/T1)) * numpy.piecewise(f, [ f<start, ( (f>=start)  & (f<(start+length)) ), 
                                                 f>=(start+length)], 
                                             [0,
                                              lambda f: (1-numpy.exp( -(f-start) / T1)), 
                                             lambda f: numpy.exp(-(f-start-length) / T1) * (1-numpy.exp(-length/T1)), 
                                             ] )
        
        @apply_along_axis()
        def guess(self, xs, fs, polarity=None, **kwargs):
            return self._guess(xs, fs, polarity=polarity, **kwargs)
        
        def findEdgeIndex(self, data, threshold, sign = +1):
                if sign >0:
                    # looking for a rising edge
                    # find where we are below threshold
                    for i1,x in enumerate(data):
                        if x < threshold:
                            break
                    # find where we are above threshold
                    for i2,x in enumerate(data[i1:]):
                        if x > threshold:
                            break        
                else:
                    # looking for a falling edge
                    # find where we are above threshold
                    for i1,x in enumerate(data):
                        if x > threshold:
                            break

                    # find where we are below threshold
                    for i2,x in enumerate(data[i1:]):
                        if x < threshold:
                            break
                            
                i = i1+i2
                return i
                #Interpolation
                #if i < len(data)-1:
                #    return i + ((data[i-1]-threshold)/(data[i-1]-data[i]))
                #else:
                #    # we fell off the end of the array
                #    return 0.0
        
        def _guess(self, xs, fs, polarity=None, **kwargs):
            '''
            Input:
                xs, fs - values of the independent and dependent variables
                polarity - expected peak polarity. defaults to auto-detection.
                f0 - estimated centre frequency. if provided, use local extremum
                    closest to f0 instead of global extremum.
                additional keyword arguments are ignored.
            '''
            # preprocess data
            if not polarity:
                polarity = 1
            fs = polarity * self.preprocess(xs, fs)
            
            #Estimate amplitude
            amplitude = numpy.max(fs)
            
            #Find postive and negative edges
            idx1 = self.findEdgeIndex(fs, amplitude/2)
            idx2 = self.findEdgeIndex(fs, amplitude/2, sign=-1)
            
            start = xs[idx1]
            length = xs[idx2] - xs[idx1]
            T1 = length/3.
            
            return OrderedDict(zip(self.PARAMETERS, (length,start,amplitude,T1))) 
