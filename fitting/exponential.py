import numpy
from collections import OrderedDict
from . import FitBase, apply_along_axis
from scipy import stats

class Exponential(FitBase):
    '''
    fit amplitude*exp(-Gamma*t)+offset to measured data
    '''
    PARAMETERS = ['Gamma', 'amplitude', 'offset']

    @staticmethod
    def f(ts, Gamma, amplitude, offset):
        '''
        offset + amplitude * exp(-Gamma*t)
        '''
        return offset+amplitude*numpy.exp(-Gamma*ts)

    @apply_along_axis()
    def guess(self, xs, fs, **kwargs):
        '''
        Input:
            xs, fs - values of the independent and dependent variables
                xs are assumed to be equally spaced
            additional keyword arguments are ignored.
        '''
        # preprocess data
        fs = self.preprocess(xs, fs)
        # for polarity, compare the first and second halves of the data points
        polarity = numpy.mean(fs[:len(fs)//2]) > numpy.mean(fs[len(fs)//2:])
        quantiles = stats.mstats.mquantiles(fs, [0.05, 0.5, 0.95])
        if polarity:
            # more points in the upper range -> positive amplitude
            offset_est = quantiles[0]
            amplitude_est = quantiles[2]-offset_est
        else:
            # more points in the lower range -> negative amplitude
            offset_est = quantiles[2]
            amplitude_est = quantiles[0]-offset_est
        # estimate that the user measured for a period of 5./Gamma
        #TODO: implement proper decay rate guesser
        Gamma_est = 5./(max(xs)-min(xs))
        #Gamma_est = -numpy.mean(numpy.diff(fs)/numpy.diff(xs)/(fs[:-1]-offset_est))/2

        return OrderedDict(zip(self.PARAMETERS, (Gamma_est,amplitude_est,offset_est))) 

class ExponentialGF(FitBase):
    '''
    fit gf T1, offset + amplitude ( gamma_e exp(- gamma_f t) - gamma_f exp(- gamma_e t) ) / (gamma_e - gamma_f)
    '''
    PARAMETERS = ['gamma_e', 'gamma_f', 'amplitude', 'offset']

    @staticmethod
    def f(ts, gamma_e, gamma_f, amplitude, offset):
        '''
        offset + amplitude ( gamma_e exp(- gamma_f t) - gamma_f exp(- gamma_e t) ) / (gamma_e - gamma_f)
        '''
        return offset+ amplitude * (gamma_e * numpy.exp(-gamma_f*ts) - gamma_f * numpy.exp(-gamma_e*ts)) / (gamma_e - gamma_f)

    @apply_along_axis()
    def guess(self, xs, fs, **kwargs):
        '''
        Input:
            xs, fs - values of the independent and dependent variables
                xs are assumed to be equally spaced
            additional keyword arguments are ignored.
        '''
        # preprocess data
        fs = self.preprocess(xs, fs)
        # for polarity, compare the first and second halves of the data points
        polarity = numpy.mean(fs[:len(fs)//2]) > numpy.mean(fs[len(fs)//2:])
        quantiles = stats.mstats.mquantiles(fs, [0.05, 0.5, 0.95])
        if polarity:
            # more points in the upper range -> positive amplitude
            offset_est = quantiles[0]
            amplitude_est = quantiles[2]-offset_est
        else:
            # more points in the lower range -> negative amplitude
            offset_est = quantiles[2]
            amplitude_est = quantiles[0]-offset_est
        # estimate that the user measured for a period of 5./Gamma
        #TODO: implement proper decay rate guesser
        Gamma_est = 5./(max(xs)-min(xs))

        return OrderedDict(zip(self.PARAMETERS, (Gamma_est*2, Gamma_est*5,amplitude_est,offset_est)))         