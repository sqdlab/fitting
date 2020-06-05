import numpy
import scipy.stats
from collections import OrderedDict
from . import FitBase, apply_along_axis

class Asymmetric_Lorentzian(FitBase):
        PARAMETERS = ['f0', 'df', 'offset', 'amplitude','A1', 'A2']
    
        @staticmethod
        def f(f, f0, df, offset, amplitude, A1, A2):
            '''
            offset + A1*f + (amplitude + A2*f) / sqrt( 1 + (2*(f-f0)/df)**2 )
            '''
            return offset+A1*f + (amplitude+A2*f) / numpy.sqrt(1.+(2.*(f-f0)/df)**2)
        
        @apply_along_axis()
        def guess(self, xs, fs, polarity=None, **kwargs):
            return self._guess(xs, fs, polarity=polarity, **kwargs)
        
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
            fs = self.preprocess(xs, fs)
            # determine noise level and peak direction
            quantiles = scipy.stats.mstats.mquantiles(fs, [0.1, 0.5, 0.9])
            if polarity is None:
                # a positive peak is indicated by the 0.1-to-0.5 distance being smaller than the 0.5-to-0.9 distance
                polarity = 1 if (numpy.diff(numpy.diff(quantiles))>0) else -1
            else:
                polarity = 1 if (polarity > 0) else -1
            offset = quantiles[0] if (polarity==1) else quantiles[-1]
            #Determine overall slope A1
            A1 = (fs[0]-fs[-1])/(xs[0]-xs[1])
            #Assume symmetric peak to start with
            A2 = 0            
            # find centre frequency
            if 'f0' in kwargs:
                f0 = kwargs['f0']
                # determine index corresponding to user f0
                f0_idx = numpy.searchsorted(xs if (xs[-1]>=xs[0]) else xs[::-1], f0)
                f0_idx = numpy.clip(f0_idx, 0, len(xs))
                # follow gradient to find local extremum close to f0_idx
                while (f0_idx>0) and (polarity*fs[f0_idx-1]>polarity*fs[f0_idx]): 
                    f0_idx = f0_idx-1
                while (f0_idx<len(xs)-2) and (polarity*fs[f0_idx+1]>polarity*fs[f0_idx]): 
                    f0_idx = f0_idx+1
            else:
                f0_idx = numpy.argmax(polarity*fs)
            f0 = xs[f0_idx]
            # calculate amplitude
            amplitude = fs[f0_idx]-offset
            # determine half width half maximum in both directions from the extremum
            df_mask = polarity*fs >= polarity*(offset+amplitude/numpy.sqrt(2))
            df_idx_r = f0_idx - 1 + numpy.count_nonzero(numpy.cumprod(df_mask[f0_idx:]))
            df_idx_l = f0_idx + 1 - numpy.count_nonzero(numpy.cumprod(df_mask[f0_idx::-1]))
            if df_idx_l == df_idx_r:
                if df_idx_l > 0: 
                    df_idx_l -= 1
                else:
                    df_idx_r += 1
            df = xs[df_idx_r]-xs[df_idx_l]
            return OrderedDict(zip(self.PARAMETERS, (f0,df,offset,amplitude, A1, A2))) 
