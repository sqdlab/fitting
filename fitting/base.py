import functools
import numpy
import scipy.optimize
import logging

take_abs = lambda xs,ys: numpy.abs(ys)
take_phase = lambda xs, ys: numpy.unwrap(numpy.angle(ys))
take_phase_p = lambda p: lambda xs, ys: numpy.unwrap(numpy.angle(numpy.exp(1.0j*p*xs*numpy.pi)*ys))
take_real= lambda xs,ys: numpy.real(ys)
take_imag = lambda xs,ys: numpy.imag(ys)
take_neg = lambda xs,ys: -ys

def apply_along_axis(multiple_outputs=False):
    '''
    decorate a function that operates on 1d vector inputs xs and fs so it can process
    arbitrary dimensional inputs
    
    for non-ndarray outputs of function, the decorated function returns a
      ndarray with dtype=object and the shape of xs with axis removed
    for ndarray outputs, the decorated function returns a
      ndarray with the dtype of the output and the shape of the output appended
    '''
    def decorator(function):
        @functools.wraps(function)
        def decorated_function(self, xs, fs, axis=None, *args, **kwargs):
            # if axis is None, operate on flattened array
            if axis is None:
                return function(self, numpy.ravel(xs), numpy.ravel(fs), *args, **kwargs)
            
            # broadcast xs and fs to the same shape
            xs, fs = numpy.broadcast_arrays(xs, fs)
            #
            resultss = None
            iter_shape = list(xs.shape)
            iter_shape.pop(axis)
            for idxs_out in numpy.ndindex(*iter_shape):
                # execute function on an 1d slice of the inputs
                idxs_in = list(idxs_out)
                if axis>=0:
                    idxs_in.insert(axis, slice(None))
                elif axis==-1:
                    idxs_in.append(slice(None))
                else:
                    idxs_in.insert(axis+1, slice(None))
                results = function(self, xs[idxs_in], fs[idxs_in], *args, **kwargs)
                # use the same output buffering code independent of multiple_outputs
                if multiple_outputs is False:
                    results = (results,)
                # create output buffers during first iteration
                if resultss is None:
                    resultss = []
                    for result in results:
                        # array outputs are wrapped up, others are stored as python objects
                        if isinstance(result, numpy.ndarray):
                            resultss.append(numpy.zeros(iter_shape+list(result.shape), dtype=result.dtype))
                        else:
                            resultss.append(numpy.zeros(iter_shape, dtype=numpy.object))
                # store results in buffers
                for idx_result in range(len(results)):
                    resultss[idx_result][idxs_out] = results[idx_result]
            # return buffered data
            return resultss if multiple_outputs else resultss[0]
        return decorated_function
    return decorator

def merge_dicts(*dicts):
    merged = dicts[0].copy()
    for dic in dicts[1:]:
        merged.update(dic)
    return merged

def interp(xs, dx):
    '''
    Add new points between the elements of xs so the distance between 
    points is lower than or equal to dx.
    
    Parameters
    ----------
    xs : `array` of `float`
        Fixed point list. Elements of xs are guaranteed to be in the
        output (up to rounding errors). Duplicate elements are ignored.
    dx : `float`
        Maximum distance between elements in the output.
        
    Returns
    -------
    interpolated : `array` of `float`
        `xs` with equally spaced points added between the elements such
        that the distance between adjacent points is at most `dx`.
    '''
    bin_sizes = numpy.ceil(numpy.abs(numpy.diff(xs) / dx))
    if numpy.all(bin_sizes == 1):
        return xs
    else:
        out_idxs = numpy.arange(numpy.sum(bin_sizes) + 1)
        fix_idxs = numpy.concatenate([(0,), numpy.cumsum(bin_sizes)])
        return numpy.interp(out_idxs, fix_idxs, xs)
    
class FitBase(object):
    '''
    Fit data to a function.
    Meant to be subclassed for each function.
    '''
    # names of the parameters returned by fit()
    PARAMETERS = []
    # true if the measurement returns more than one set of 
    # parameters for each trace fit
    RETURNS_MULTIPLE_PARAMETER_SETS = False
    COMPLEX_INPUT = False
    
    def __init__(self, preprocess=None, guess={}, fixed={}):
        '''
        Create a new fitter.
        
        Input:
            preprocess - fs=f(xs, fs) is called by the guessers and fitters
                to pre-process the data before fitting
            guess - default guesses used by fit()
            fixed - default fixed parameters used by fit()
        '''
        if preprocess is None:
            self.preprocess = lambda xs, ys: ys
        else:
            self.preprocess = preprocess
        self.default_guess = guess
        self.default_fixed = fixed
    
    @apply_along_axis(False)
    def guess(self, xs, fs, **kwargs):
        '''
        guess starting parameters for the fit of f(xs)==fs
        
        Input:
            xs, fs, axis - (see fit)
        Returns:
            OrderedDict of guessed parameters that can be 
                passed to fit() as **kwargs
                or to f() as *args
        '''
        return {}
        
    @apply_along_axis(True)
    def fit(self, xs, fs, guess={}, fixed={}, weights=1., **kwargs):
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
            weights - weight of the observations
        Note:
            The objective function is \sum |w_i| |f_i - f(x_i)|
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
            return numpy.abs(weights)*numpy.abs(self.f(xs, *[params[k] for k in self.PARAMETERS]) - fs)

        res = scipy.optimize.leastsq(function, p0s, full_output=1)
        (p_opt, p_cov, _, errmsg, ier) = res

        if ier not in [1,2,3,4]:
            logging.info(__name__ + ': fit failed with message {0:s}.'.format(errmsg))
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
    
        return p_opt, p_cov
    
    @apply_along_axis()
    def plot(self, xs, fs, guess={}, fixed={}, **kwargs):
        '''
        plot data, guess and fit result using matplotlib
        
        Input:
            plt - matplotlib.pyplot
            xs, fs - values of the dependent and independent variables
        '''
        p_0 = self.guess(xs, fs, **merge_dicts(self.default_guess, guess, 
                                               self.default_fixed, fixed)).values()
        p_opt, _ = self.fit(xs, fs, guess=guess, fixed=fixed, **kwargs)
        import matplotlib.pyplot as plt
        if self.COMPLEX_INPUT:
            fig = plt.figure(figsize=(12,6))
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            axs = ((numpy.real, ax1), (numpy.imag, ax2))
        else:
            fig = plt.figure(figsize=(6,6))
            ax1 = fig.add_subplot(111)
            axs = ((numpy.real, ax1),)
        xs_fit = interp(xs, (numpy.max(xs)-numpy.min(xs))/251.)
        for cast, ax in axs:
            ax.plot(xs, cast(self.preprocess(xs, fs)), '.', label='data')
            if len(p_0) == len(self.PARAMETERS):
                ax.plot(xs_fit, cast(self.f(xs_fit, *p_0)), '--', label='guess')
            ax.plot(xs_fit, cast(self.f(xs_fit, *p_opt)), '-', label='fit')
            ax.legend(loc='best')
        plt.close(fig)
        return fig
