import numpy
import scipy.stats.mstats
from collections import OrderedDict
import logging
from . import FitBase, apply_along_axis

class Oscillation(FitBase):
    '''
    fit amplitude*exp(-gamma*t)*cos(omega*t+phi)+offset to measured data
    '''
    PARAMETERS = ['f0', 'phi', 'Gamma', 'amplitude', 'offset']

    @staticmethod
    def f(ts, f0, phi, Gamma, amplitude, offset):
        '''
        offset + amplitude * exp(-Gamma*t) * cos(2*pi*f0*t+phi)
        '''
        return offset-amplitude*numpy.exp(-Gamma*ts)*numpy.cos(2*numpy.pi*f0*ts+phi)


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
        # basic guess of amplitude and offset
        amplitude_est = -(numpy.max(fs)-numpy.min(fs))/2.
        offset_est = numpy.mean(fs)
        
        # check if the signal is balanced around the mean (experimental)
        if False:
            quantiles = scipy.stats.mstats.mquantiles(fs, [0.05, 0.5, 0.95])
            mean = numpy.mean(fs)
            amplitude_est = (numpy.max(fs)-numpy.min(fs))/2.
            amplitude_est = (quantiles[2]-quantiles[0])/2.
            if abs(quantiles[1]-mean) < .1*amplitude_est:
                offset_est = mean
            else:
                amplitude_est *= 2
                offset_est = quantiles[0] if (quantiles[1] < mean) else quantiles[2]

        # guess frequency and phase from fft
        fftlen = 1*len(xs)
        time_spacing = numpy.mean(numpy.diff(xs))
        freqs = numpy.fft.fftfreq(fftlen, time_spacing)
        freq_spacing = freqs[1]-freqs[0]
        fc = numpy.fft.fft(fs-offset_est, fftlen) # removes DC component from FFT
        fa = numpy.abs(fc)
        peak_idx = numpy.argmax(fa) 
        
        f0_est = freqs[peak_idx]
        phi_est = numpy.angle(fc[peak_idx])
        
        # estimate decay rate from with of Lorentzian, default to 20% of the recorded window
        peak_max = fa[peak_idx]
        # mean of fft amplitudes of the two points adjacent to the maximum.
        peak_off = (
            ((fa[peak_idx-1] if peak_idx>0 else 0) + (fa[peak_idx+1] if peak_idx<len(fa)-1 else 0)) / 
            ((1. if peak_idx>0 else 0.) + (1. if peak_idx<len(fa)-1 else 0.))
        )
        Gamma_est = freq_spacing/numpy.sqrt(numpy.abs(peak_max/peak_off)-1.) if (peak_off>0 and peak_off<peak_max) else (.2/(xs[-1]-xs[0]))

        return OrderedDict(zip(Oscillation.PARAMETERS, (f0_est,phi_est,Gamma_est,amplitude_est,offset_est))) 



class ComplexOscillation(Oscillation):
    '''
    fit 0.5*((I0+I1)+1j*(Q0+Q1)) - 0.5*((I1-I0)+1j*(Q1-Q0))*exp(-Gamma*t)*cos(2*pi*f0*t+phi) to measured data
    '''
    PARAMETERS = ['f0', 'phi', 'Gamma', 'I0', 'Q0', 'I1', 'Q1']
    COMPLEX_INPUT = True

    @staticmethod
    def f(ts, f0, phi, Gamma, I0, Q0, I1, Q1):
        '''
        0.5*((I0+I1)+1j*(Q0+Q1)) - 0.5*((I1-I0)+1j*(Q1-Q0))*exp(-Gamma*t)*cos(2*pi*f0*t+phi)
        '''
        offset = 0.5*(I0+I1)+0.5j*(Q0+Q1)
        amplitude = 0.5*(I1-I0)+0.5j*(Q1-Q0)
        return offset-amplitude*numpy.exp(-Gamma*ts)*numpy.cos(2*numpy.pi*f0*ts+phi)

    @apply_along_axis()
    def guess(self, xs, fs, **kwargs):
        # preprocess data
        fs = self.preprocess(xs, fs)
        # warn of real inputs
        if not numpy.any(numpy.iscomplex(fs)):
            logging.warning(__name__+': use Oscillation for better results with real inputs.')
            
        kwargs_re = {}
        kwargs_im = {}
        if ('I0' in kwargs) and ('I1' in kwargs):
            kwargs_re['amplitude'] = kwargs['I1']-kwargs['I0']
            kwargs_re['offset'] = 0.5*(kwargs['I1']+kwargs['I0'])
            kwargs.pop('I0')
            kwargs.pop('Q0')
        kwargs_re.update(kwargs)

        if ('Q0' in kwargs) and ('Q1' in kwargs):
            kwargs_im['amplitude'] = kwargs['Q1']-kwargs['Q0']
            kwargs_im['offset'] = 0.5*(kwargs['Q1']+kwargs['Q0'])
            kwargs.pop('I0')
            kwargs.pop('Q0')
        kwargs_im.update(kwargs)
        
        p_est_re = super(ComplexOscillation, self).guess(xs, numpy.real(fs), **kwargs_re)
        p_est_im = super(ComplexOscillation, self).guess(xs, numpy.imag(fs), **kwargs_im)
        for p_est in (p_est_re, p_est_im):
            if abs(p_est['phi']) > numpy.pi/2.:
                p_est['amplitude'] = -p_est['amplitude']
                p_est['phi'] = (p_est['phi']+2*numpy.pi)%(2*numpy.pi)-numpy.pi
        
        weights = numpy.array((numpy.abs(p_est_re['amplitude']), numpy.abs(p_est_im['amplitude'])))
        weights /= numpy.sum(weights)
        f0_est = p_est_re['f0']*weights[0]+p_est_im['f0']*weights[1]
        phi_est = p_est_re['phi']*weights[0]+p_est_im['phi']*weights[1]
        Gamma_est = p_est_re['Gamma']*weights[0]+p_est_im['Gamma']*weights[1]
        
        if ('phi' in kwargs) and (kwargs['phi'] % numpy.pi == 0.):
            IQ0_est = fs[0]
            IQ1_est = fs[numpy.argmax(abs(fs-IQ0_est))]
        elif ('phi' in kwargs) and (kwargs['phi'] % numpy.pi == numpy.pi/2.):
            IQ1_est = fs[0]
            IQ0_est = fs[numpy.argmax(abs(fs-IQ1_est))]
        else:
            IQ0_est = (p_est_re['offset']-p_est_re['amplitude']) + 1j*(p_est_im['offset']-p_est_im['amplitude'])
            IQ1_est = (p_est_re['offset']+p_est_re['amplitude']) + 1j*(p_est_im['offset']+p_est_im['amplitude'])
        I0_est = numpy.real(IQ0_est)
        Q0_est = numpy.imag(IQ0_est)
        I1_est = numpy.real(IQ1_est)
        Q1_est = numpy.imag(IQ1_est)
        
        return OrderedDict(zip(ComplexOscillation.PARAMETERS, (f0_est,phi_est,Gamma_est,I0_est,Q0_est,I1_est,Q1_est))) 