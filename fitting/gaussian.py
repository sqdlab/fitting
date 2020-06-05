import numpy
from . import Lorentzian, apply_along_axis

class Gaussian(Lorentzian):
        PARAMETERS = ['f0', 'df', 'offset', 'amplitude']
    
        @staticmethod
        def f(f, f0, df, offset, amplitude):
            '''
            offset + amplitude * exp( -(f-f0)**2 / (df/2)**2 )
            '''
            return offset+amplitude*numpy.exp(-(f-f0)**2/(df/2.)**2)
        
        @apply_along_axis()
        def guess(self, xs, fs, polarity=None, **kwargs):
            ''' uses the lorentzian guesser for start parameters'''
            p0 = self._guess(xs, fs, polarity=polarity, **kwargs)
            p0['df'] = p0['df']*numpy.sqrt(1./numpy.log(2))
            return p0
