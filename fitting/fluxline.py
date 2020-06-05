import numpy
import scipy.stats.mstats
from collections import OrderedDict
import logging
from . import FitBase, apply_along_axis
from .base import merge_dicts

class FluxLineCal(FitBase):
    '''
	This class fits the given data with the analytical expression of the probability for the qubit
	to be in the excited state when the qubit is under a drive and an oscillating flux, under two rotating waves approximations. See the Reports about flux control for further information.
	
	The measured data is fitted with the model:
	amplitude/8[4 
				- cos(t(pi*(a0 * asw) - wr)) 
				- cos(t(pi*(a0 * asw) + wr))
				- 4cos(phi0 + phisw) cos(phi0 + phisw + wr*t)
				+ 2cos(pi*(a0*asw)t) cos(2phi0 + 2phisw + wr*t)
				]
	+ offset
	where a0 and phi0 are the intrinsic parameter of the flux line we want to measure, and af and phisw
	the parameters we can sweep. wR is the frequency of the Rabi oscillations, and "amplitude" and "offset"
	are	adjustment parameters.

    give rabi frequency 'wr' and sweep paramters amplitude 'asw' and phase 'phis'
    '''
    PARAMETERS = [ 'wr', 't1', 'phis' , 'phi0', 'asw' , 'a0', 'amplitude', 'offset']

    @staticmethod
    def f(ts, wr, t1, phis, phi0, asw, a0, amplitude, offset, resErr = 0):
        result = offset + numpy.exp(-ts/t1)*amplitude*0.125*(4- numpy.cos(ts*(numpy.pi*(asw*a0)- wr))- numpy.cos(ts*(numpy.pi*(asw*a0) + wr))- 4*numpy.cos(phi0+phis)*numpy.cos(phi0+phis + ts*wr)+ 2*numpy.cos(numpy.pi*asw*a0*ts) * numpy.cos(2*(phi0+phis)+ts*wr))        
        return result
 


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
        amplitude_est = numpy.abs(numpy.max(fs)-numpy.min(fs))
        offset_est = numpy.mean(fs)
        
        # check if the signal is balanced around the mean (experimental)
        if False:
            quantiles = scipy.stats.mstats.mquantiles(fs, [0.05, 0.5, 0.95])
            mean = numpy.mean(fs)
            amplitude_est = (numpy.max(fs)-numpy.min(fs))
            amplitude_est = (quantiles[2]-quantiles[0])
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

        #return OrderedDict(zip(Oscillation.PARAMETERS, (f0_est,phi_est,amplitude_est,offset_est))) 
        return {'wr': 2*numpy.pi*numpy.abs(f0_est), 'phi0': phi_est, 'a0': 1, 'amplitude': amplitude_est, 'offset': offset_est-amplitude_est/2, 't1': 2e-6}
        
    @apply_along_axis(True)
    def fit(self, xs, fs, guess={}, fixed={}, minimizationMethod = 'Powell', **kwargs):
        '''
        fit values to function
        
        Input:
            xs, fs - values of the dependent and independent variables.
            axis - axis from which to take the input vector. all other dimensions will
                be preserved. default is to operate on the flattened inputs.
            guess - items are passed to the parameter guesser and at the same time take 
                precedence over guessed parameters when determining initial values
            fixed - items are treated as constants and are not optimized
                guess and fixed are merged with the guess and fixed parameters passed to
                the constructor and take precedence over those
        Returns:
            (ndarrays of:)
            optimized parameters
            covariance matrix
        '''
        # merge default guess/fixed with guess/fixed arguments
        guess = merge_dicts(self.default_guess, guess)
        fixed = merge_dicts(self.default_fixed, fixed)
        # old-style guesses
        if len(kwargs):
            logging.warning('passing guesses as **kwargs is deprecated. use guess argument instead.')
            guess.update(kwargs)
        # calculate starting parameters (user-provided parameters take precedence)
        try:
            if numpy.all([k in guess for k in self.PARAMETERS]):
                p0 = {}
            else:
                p0 = self.guess(xs, fs, **merge_dicts(guess, fixed))
            p0.update((k,v) for k,v in guess.items() if k in self.PARAMETERS)
        except:
            logging.warning(__name__ + ': parameter guesser raised an exception. ' + 
                'using all 1s as starting parameters.')
            p0 = {}

        # fit data
        fs = self.preprocess(xs, fs)

        pks = [k for k in self.PARAMETERS if k not in fixed]
        p0s = [(p0[k] if k in p0 else 1.)
               for k in self.PARAMETERS 
               if k not in fixed]

        def function(ps):
            ''' 
            leastsq objective function supporting complex inputs 
            
            Input:
                list of fit parameters
            '''
            params = dict(zip(pks, ps))
            params.update(fixed)
            return numpy.sum(numpy.abs(self.f(xs, *[params[k] for k in self.PARAMETERS]) - fs))
        
        '''   
        def resErr(p_opt, measured_data):
            
            #This function takes the parameters optimized by the fit, p_opt, and returns the sum of the square of their difference with the measured data.
            #This gives a criterion to compare the respective relevance of two fitting results.
            
            return residualError
        '''
        
        res = scipy.optimize.minimize(function, p0s, method=minimizationMethod)
        p_opt = res.x
        resErr = res.fun
        p_cov = None

        if not res.success:
            logging.info(__name__ + ': fit failed with message {0:s}.'.format(res.message))
            p_opt = numpy.NaN*numpy.ones((len(p0s),))
            p_cov = numpy.NaN*numpy.ones((len(p0s),len(p0s)))
        else:
            if (len(fs) > len(p0)) and p_cov is not None:
                s_sq = (function(p_opt)**2).sum()/(len(fs)-len(p0))
                p_cov = p_cov * s_sq
            else:
                p_cov = numpy.inf*numpy.ones((len(p_opt),len(p_opt)))

        if len(fixed):
            # add fixed elements to p_opt
            p_opt = list(p_opt)
            for idx, pk in enumerate(self.PARAMETERS):
                if pk in fixed:
                    p_opt.insert(idx, fixed[pk])
            p_opt = numpy.array(p_opt)
                    
            if p_cov is not None:
                # blow covariance matrix up
                p_cov_new = numpy.inf*numpy.ones((len(self.PARAMETERS),len(self.PARAMETERS)))
                pmap = [self.PARAMETERS.index(pk) for pk in pks]
                for sidx0, didx0 in enumerate(pmap):
                    for sidx1, didx1 in enumerate(pmap):
                        p_cov_new[didx0,didx1] = p_cov[sidx0,sidx1] 
                p_cov = p_cov_new
    #EDIT 01/12/2015 : added function(p_opt) in the return. Does not seems to work, though...
        return p_opt, p_cov, resErr