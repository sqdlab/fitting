import numpy
import scipy.signal
import scipy.stats
import scipy.optimize
import logging

from . import FitBase, apply_along_axis

class PeakFind(FitBase):
    PARAMETERS = ['f0', 'df', 'offset', 'amplitude']
    
    def __init__(self, peak_limit=0, polarity=1, peak_args={}, **kwargs):
        '''
        Input:
            peak_limit (optional) - maximum number of peaks to process, defaults to 0=no limit,
                largest amplitude peaks are processed first
            polarity - +1 for positive peaks, -1 for negative peaks
            peak_args - arguments passed to peak finder (modified find_peaks_cwt), default
                {'widths': logspace(5, 8, base=10.),
                 'min_snr': 5,
                 'noise_perc': 90,
                 'gap_thresh': 5, 
                 'max_distance_exp': 0.,
                 'max_distance_factor': 3. }
                If not provided, max_distance is calculated according to 
                max_distance_factor * (widths / freq_spacing)**max_distance_exp.
                The default noise_perc of 90% assumes narrow peaks in a wide
                measurement window, i.e. the lower 90% of samples are considered
                noise.
            (see FitBase for additional options)
        '''
        super(PeakFind, self).__init__(**kwargs)
        self.peak_limit = peak_limit
        self.polarity = polarity
        self.peak_args = {
            'wavelet': scipy.signal.ricker,
            'widths': numpy.logspace(5, 8, base=10.),
            'min_snr': 5,
            'noise_perc': 90,
            'gap_thresh': 5,
            'min_length': None,
            'max_distance_exp': 0.,
            'max_distance_factor': 3.
        }
        self.peak_args.update(peak_args)
        self.RETURNS_MULTIPLE_PARAMETER_SETS = (peak_limit != 1)

    @staticmethod
    def f(f, *args):
        '''
        offset + \sum_i{ amplitude_i / sqrt( 1 + (2*(f-f0_i)/df_i)**2 ) }

        Input, either:
          (- one or multiple dictionaries with keys PeakFind.PARAMETERS
              only the last instance of offset will be used)
          - one or multiple lists or tuples of (f0, df, amplitude, offset)
              only the last instance of offset will be used
          - one or multiple repetitions of f0, df, amplitude, 
              followed by a single instance of offset
        '''
        if not len(args):
            return 0.
        result = 0.
        if numpy.all([isinstance(arg, dict) for arg in args]):
            for parameters in args:
                f0, df, amplitude = (parameters[key] for key in ('f0', 'df', 'amplitude'))
                result += amplitude/numpy.sqrt(1.+(2.*(f-f0)/df)**2)
            result += parameters['offset']
        elif numpy.all([numpy.iterable(arg) for arg in args]):
            for parameters in args:
                f0, df, amplitude = (parameters[idx] for idx in (0, 1, 3))
                result += amplitude/numpy.sqrt(1.+(2.*(f-f0)/df)**2)
            result += parameters[2]
        else:
            for idx in range(0, len(args)-1, 3):
                f0, df, amplitude = args[idx:(idx+3)]
                result += amplitude/numpy.sqrt(1.+(2.*(f-f0)/df)**2)
            result += args[-1]
        return result

    @apply_along_axis(multiple_outputs=True)
    def fit(self, xs, fs, plt=None, **kwargs):
        '''
        Find peaks in data and fit Lorentzians to them.
        
        Input:
            xs, fs - values of the independent and dependent variables
            plt - reference to matplotlib.pyplot for debug plotting
            polarity - +1 for positive peaks, -1 for negative peaks
        Returns:
            list of parameter dictionaries for each Lorentzian as returned by Lorentzian
            list of corresponding covariance matrices
        '''
        # preprocess data
        polarity = kwargs.get('polarity', self.polarity)
        fs = polarity*self.preprocess(xs, fs)
        
        peak_args = self.peak_args.copy()
        # normalize widths, assuming equal spacing of frequency points
        freq_spacing = numpy.mean(numpy.diff(xs))

        wavelet = peak_args['wavelet']
        widths = peak_args['widths']/freq_spacing
        #widths = widths[widths>=1]
        max_distances = peak_args.get('max_distances', peak_args['widths']/2.0)/freq_spacing
        max_distances = peak_args.get('max_distance_factor')*max_distances**peak_args.get('max_distance_exp') 
        max_distances = numpy.where(max_distances>=1., max_distances, 1)
        gap_thresh = peak_args['gap_thresh']
        min_length = peak_args['min_length']
        min_snr = peak_args['min_snr']
        noise_perc = peak_args['noise_perc']
        # minimum width 1
        max_distances = max_distances[widths>=1]
        widths = widths[widths>=1]
        
        # find approximate peak frequencies
        cwt = scipy.signal.cwt(fs, wavelet, 2*numpy.pi*widths)
        rls_raw = scipy.signal._peak_finding._identify_ridge_lines(cwt, max_distances=max_distances, gap_thresh=gap_thresh)
        logging.debug(__name__ + ': initial ridge lines: ' + str(len(rls_raw)))
        rls = PeakFind._filter_ridge_lines(cwt, rls_raw, min_length=min_length, min_snr=min_snr, noise_perc=noise_perc)
        logging.debug(__name__ + ': filtered ridge lines: ' + str(len(rls)))
        # sort by frequency
        rls = sorted(rls, key=lambda a: int(a[1][0]))
        # calculate position, width and snr for each peak
        peak_if0s = numpy.array([line[1][0] for line in rls])
        peak_idfs = numpy.array([PeakFind._ridge_line_width(cwt, line, widths) for line in rls])
        peak_snrs = numpy.array([PeakFind._ridge_line_snr(cwt, line, noise_perc) for line in rls])
        # observe peak_limit
        if self.peak_limit and len(rls)>self.peak_limit:
            filter_idxs = numpy.sort(numpy.argsort(peak_snrs)[-self.peak_limit:])
            peak_if0s = peak_if0s[filter_idxs]
            peak_idfs = peak_idfs[filter_idxs]
            peak_snrs = peak_snrs[filter_idxs]
        logging.debug(__name__+': identified peaks: '+str(list(zip(peak_if0s, peak_idfs, peak_snrs))))

        # plot ridge lines
        if plt is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.pcolor(xs, widths, cwt)
            for rl in rls_raw:
                ax.plot(xs[rl[1]], widths[rl[0]], lw=(3 if rl[1][0] in peak_if0s else 1))
        # plot snr graphs
        if plt is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            for rl in rls:
                plt.plot(widths[rl[0]], cwt[rl[0], rl[1]])
            
        # convert edge tuples into slices
        to_slice = lambda extent: slice(max(0,int(round(extent[0]))), min(len(xs),int(round(extent[1]))))
        
        # fit Lorentzians to all peaks
        # strategy for multiple peaks:
        # a) if peaks are well separated (say >5dfs), fit individually -- faster and more robust
        # b) if peak distance is in (df, 5dfs), fit sum of lorentzians
        # c) if peak distance is <df, fit the widest peak only
        
        # allocate output buffers
        p0s = [] # for debugging only
        p_opts = []
        p_covs = []
        
        # step 0: calculate global offset
        quantiles = scipy.stats.mstats.mquantiles(fs, [0.1, 0.5, 0.9])
        offset = quantiles[0] 
        
        # step 1: split well-separated peaks into subproblems
        groups, extents = PeakFind.group_peaks(peak_if0s, peak_idfs, sigmas=5)
        for group, extent in zip(groups, extents):
            extent_sl = to_slice(extent)
            #print group, extent_sl

            # step 2: combine groups of close peaks into single peaks
            group_if0s = []
            group_idfs = []
            subgroups, subextents = PeakFind.group_peaks(peak_if0s[group], peak_idfs[group], sigmas=1)
            #print 'subgroups', subgroups, subextents
            for subgroup, subextent in zip(subgroups, subextents):
                # grouping method 0: take widest peak
                subgroup_dominus = numpy.argmax(peak_idfs[group][subgroup])
                group_if0s.append(peak_if0s[group][subgroup][subgroup_dominus])
                group_idfs.append(peak_idfs[group][subgroup][subgroup_dominus])
                # grouping method 1: take central frequency & combined widths
                #group_if0s.append(numpy.mean(peak_if0s[subgroup]))
                #group_idfs.append(subextent[-1]-subextent[0])

            # step 3: fit sum of Lorentzians
            # step 3a: find initial fit parameters
            p0 = []
            for if0, idf, subextent in zip(group_if0s, group_idfs, subextents):
                f0 = xs[if0]
                df = freq_spacing*idf
                amplitude = numpy.max(fs[to_slice(subextent)])-offset
                p0.extend((f0, df, amplitude))
            #offset = (numpy.sum(fs[extent_sl])-numpy.sum(self.f(xs[extent_sl], *(p0+[0.]))))/(extent_sl.stop-extent_sl.start)
            p0.append(offset)
            #print 'starting parameters', p0
            p0s.append(p0)
            # step 3b: run fit routine
            try:
                p_opt, p_cov = scipy.optimize.curve_fit(self.f, xs[extent_sl], fs[extent_sl], p0 = p0)
            except RuntimeError:
                p_opt = None # numpy.NaN*numpy.ones((len(self.PARAMETERS),))
                p_cov = None
            if(not isinstance(p_cov, numpy.ndarray)) or (p_cov.shape != (len(p0),len(p0))):
                p_cov = numpy.Inf*numpy.ones((len(p0),len(p0)))
            # arrange parameters into the output structures
            if p_opt is not None:
                for idx in range(0, len(p_opt)-1, 3):
                    idxs = [idx, idx+1, -1, idx+2]
                    p_opts.append(p_opt[idxs])
                    p_covs.append(p_cov[idxs, :][:, idxs])
        
        # plot fitted curve
        if plt is not None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.plot(xs, fs)
            for p0 in p0s:
                plt.plot(xs, self.f(xs, *p0), '--')
            if len(p_opts):
                plt.plot(xs, self.f(xs, *p_opts))
            plt.show()
        
        # we're done!
        return p_opts, p_covs

    def plot_series(self, ps, xs, p_opts, axis=-1):
        '''
        Input:
            ps - parameter value for each point
            xs - frequency value for each point
            p_opts - function parameter tuples for every series parameter value
            axis - index of the frequency axis in xs
        '''
        shape = list(xs.shape)
        shape[axis] = 1
        fs = numpy.zeros_like(xs)
        peaks = [[], []]
        for idx in numpy.ndindex(*shape):
            idx_xs = list(idx)
            idx_xs[axis] = slice(None)
            idx_xs = tuple(idx_xs)
            idx_ps = list(idx)
            idx_ps.pop(axis)
            idx_ps = tuple(idx_ps)
            fs[idx_xs] = self.f(xs[idx_xs], *p_opts[idx_ps])
            for p_opt in p_opts[idx_ps]:
                peaks[0].append(ps[idx_ps])
                peaks[1].append(p_opt[0])

        # blow x vector up
        def subs(xs, idx, sub):
            result=list(xs)
            result[idx]=sub
            return tuple(result)
        xedge = numpy.zeros(subs(xs.shape, axis, xs.shape[axis]+1))
        allslice = [slice(None)]*len(xs.shape)
        xedge[subs(allslice, axis, slice(None, -1))] = xs
        xedge[subs(allslice, axis, -1)] += xs[subs(allslice, axis, -1)]
        xedge[subs(allslice, axis, 0)] += xs[subs(allslice, axis, 0)]
        xedge[subs(allslice, axis, slice(1, None))] += xs
        xedge /= 2.
        
        # reproduce experimental data
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.close(fig)
        ax = fig.add_subplot(111)
        ax.pcolormesh(ps, xedge, fs)
        ax.plot(peaks[0], peaks[1], '.')
        return fig

    @staticmethod
    def group_peaks(if0s, idfs, sigmas=1):
        '''
        Sort peaks into disjoint groups
            
        Input:
            if0s, idfs - peak positions and widths
            sigmas - peaks within sigmas line widths are sorted into the same group
        Returns:
            groups - [[index, index, ...], ...] indices of the peaks beloning to each group
            extents - [(left edge, right edge), ...] frequency interval of each group
        '''
        # ins, outs, inouts: [(in_or_out, peak index, left/right edge position)]
        ins = [(True, idx, if0s[idx]-sigmas*idfs[idx]/2) for idx in range(len(if0s))]
        outs = [(False, idx, if0s[idx]+sigmas*idfs[idx]/2) for idx in range(len(if0s))]
        inouts = sorted(ins+outs, key=lambda a: int(a[2]))
        groups = []
        extents = []
        group = []
        status = []
        for action, idx, pos in inouts:
            if action:
                # entering range of a peak
                status.append(idx)
                group.append(idx)
            else:
                # leaving range of a peak
                status.remove(idx)
            if len(status) == 0:
                # last peak left the range
                groups.append(group)
                extents.append((ins[group[0]][2], pos+1))
                group = []
        return groups, extents

    @staticmethod
    def find_peaks_cwt(vector, widths, wavelet=None, max_distances=None, gap_thresh=None,
                       window_size=None, min_length=None, min_snr=1, noise_perc=95):
        """
        Attempt to find the peaks in a 1-D array.
        
        The general approach is to smooth `vector` by convolving it with
        `wavelet(width)` for each width in `widths`. Relative maxima which
        appear at enough length scales, and with sufficiently high SNR, are
        accepted.
        
        Parameters
        ----------
        vector : ndarray
            1-D array in which to find the peaks.
        widths : sequence
            1-D array of widths to use for calculating the CWT matrix. In general,
            this range should cover the expected width of peaks of interest.
        wavelet : callable, optional
            Should take a single variable and return a 1-D array to convolve
            with `vector`. Should be normalized to unit area.
            Default is the ricker wavelet.
        max_distances : ndarray, optional
            At each row, a ridge line is only connected if the relative max at
            row[n] is within ``max_distances[n]`` from the relative max at
            ``row[n+1]``. Default value is ``widths/2``.
        gap_thresh : float, optional
            If a relative maximum is not found within `max_distances`,
            there will be a gap. A ridge line is discontinued if there are more
            than `gap_thresh` points without connecting a new relative maximum.
            Default is 3.
        min_length : int, optional
            Minimum length a ridge line needs to be acceptable.
            Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
        min_snr : float, optional
            Minimum SNR ratio. Default 1. The signal is the value of
            the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
            noise is the `noise_perc`th percentile of datapoints.
        noise_perc : float, optional
            When calculating the noise floor, percentile of data points
            examined below which to consider noise. Calculated using
            `stats.scoreatpercentile`. Default is 10.
        
        Returns
        -------
        
        
        See Also
        --------
        tweaked version of scipy.signal.find_peaks_cwt
        
        References
        ----------
        .. [1] Bioinformatics (2006) 22 (17): 2059-2065.
        doi: 10.1093/bioinformatics/btl355
        http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
        
        """
        if gap_thresh is None:
            gap_thresh = 3 #numpy.ceil(widths[0])
        if max_distances is None:
            max_distances = widths / 2.0
        if wavelet is None:
            wavelet = scipy.signal.ricker
    
        cwt_dat = scipy.signal.cwt(vector, wavelet, widths)
        ridge_lines = PeakFind._identify_ridge_lines(cwt_dat, max_distances, gap_thresh)
        filtered = PeakFind._filter_ridge_lines(cwt_dat, ridge_lines, window_size=window_size, min_length=min_length,
                                       min_snr=min_snr, noise_perc=noise_perc)
        max_locs = [x[1][0] for x in filtered]
        return sorted(max_locs)
    
    @staticmethod
    def _boolrelextrema(data, comparator,
                  axis=0, order=1, mode='clip'):
        """
        Calculate the relative extrema of `data`.
        
        Relative extrema are calculated by finding locations where
        ``comparator(data[n], data[n+1:n+order+1])`` is True.
        
        Parameters
        ----------
        data : ndarray
            Array in which to find the relative extrema.
        comparator : callable
            Function to use to compare two data points.
            Should take 2 numbers as arguments.
        axis : int, optional
            Axis over which to select from `data`. Default is 0.
        order : int, optional
            How many points on each side to use for the comparison
            to consider ``comparator(n,n+x)`` to be True.
        mode : str, optional
            How the edges of the vector are treated. 'wrap' (wrap around) or
            'clip' (treat overflow as the same as the last (or first) element).
            Default 'clip'. See numpy.take
        
        Returns
        -------
        extrema : ndarray
            Boolean array of the same shape as `data` that is True at an extrema,
            False otherwise.
        
        See also
        --------
        argrelmax, argrelmin
        
        Examples
        --------
        >>> testdata = np.array([1,2,3,2,1])
        >>> _boolrelextrema(testdata, np.greater, axis=0)
        array([False, False, True, False, False], dtype=bool)
        
        """
        if((int(order) != order) or (order < 1)):
            raise ValueError('Order must be an int >= 1')
    
        datalen = data.shape[axis]
        locs = numpy.arange(0, datalen)
    
        results = numpy.ones(data.shape, dtype=bool)
        main = data.take(locs, axis=axis, mode=mode)
        for shift in range(1, order + 1):
            plus = data.take(locs + shift, axis=axis, mode=mode)
            minus = data.take(locs - shift, axis=axis, mode=mode)
            results &= comparator(main, plus)
            results &= comparator(main, minus)
            if(~results.any()):
                return results
        return results

    @staticmethod
    def _identify_ridge_lines(matr, max_distances, gap_thresh):
        """
        Identify ridges in the 2-D matrix.
        
        Expect that the width of the wavelet feature increases with increasing row
        number.
        
        Parameters
        ----------
        matr : 2-D ndarray
            Matrix in which to identify ridge lines.
        max_distances : 1-D sequence
            At each row, a ridge line is only connected
            if the relative max at row[n] is within
            `max_distances`[n] from the relative max at row[n+1].
        gap_thresh : int
            If a relative maximum is not found within `max_distances`,
            there will be a gap. A ridge line is discontinued if
            there are more than `gap_thresh` points without connecting
            a new relative maximum.
        
        Returns
        -------
        ridge_lines : tuple
            Tuple of 2 1-D sequences. `ridge_lines`[ii][0] are the rows of the ii-th
            ridge-line, `ridge_lines`[ii][1] are the columns. Empty if none found.
            Each ridge-line will be sorted by row (increasing), but the order
            of the ridge lines is not specified.
        
        References
        ----------
        Bioinformatics (2006) 22 (17): 2059-2065.
        doi: 10.1093/bioinformatics/btl355
        http://bioinformatics.oxfordjournals.org/content/22/17/2059.long
        
        Examples
        --------
        >>> data = np.random.rand(5,5)
        >>> ridge_lines = _identify_ridge_lines(data, 1, 1)
        
        Notes
        -----
        This function is intended to be used in conjunction with `cwt`
        as part of `find_peaks_cwt`.
        
        """
        if(len(max_distances) < matr.shape[0]):
            raise ValueError('Max_distances must have at least as many rows as matr')
    
        all_max_cols = PeakFind._boolrelextrema(matr, numpy.greater, axis=1, order=1)
        #Highest row for which there are any relative maxima
        has_relmax = numpy.where(all_max_cols.any(axis=1))[0]
        if(len(has_relmax) == 0):
            return []
        start_row = has_relmax[-1]
        #Each ridge line is a 3-tuple:
        #rows, cols,Gap number
        ridge_lines = [[[start_row],
                       [col],
                       0] for col in numpy.where(all_max_cols[start_row])[0]]
        final_lines = []
        rows = numpy.arange(start_row - 1, -1, -1)
        cols = numpy.arange(0, matr.shape[1])
        for row in rows:
            this_max_cols = cols[all_max_cols[row]]
    
            #Increment gap number of each line,
            #set it to zero later if appropriate
            for line in ridge_lines:
                line[2] += 1
    
            #XXX These should always be all_max_cols[row]
            #But the order might be different. Might be an efficiency gain
            #to make sure the order is the same and avoid this iteration
            prev_ridge_cols = numpy.array([line[1][-1] for line in ridge_lines])
            #Look through every relative maximum found at current row
            #Attempt to connect them with existing ridge lines.
            for ind, col in enumerate(this_max_cols):
                """
    If there is a previous ridge line within
    the max_distance to connect to, do so.
    Otherwise start a new one.
    """
                line = None
                if(len(prev_ridge_cols) > 0):
                    diffs = numpy.abs(col - prev_ridge_cols)
                    closest = numpy.argmin(diffs)
                    if diffs[closest] <= max_distances[row]:
                        line = ridge_lines[closest]
                if(line is not None):
                    #Found a point close enough, extend current ridge line
                    line[1].append(col)
                    line[0].append(row)
                    line[2] = 0
                else:
                    new_line = [[row],
                                [col],
                                0]
                    ridge_lines.append(new_line)
    
            #Remove the ridge lines with gap_number too high
            #XXX Modifying a list while iterating over it.
            #Should be safe, since we iterate backwards, but
            #still tacky.
            for ind in range(len(ridge_lines) - 1, -1, -1):
                line = ridge_lines[ind]
                if line[2] > gap_thresh:
                    final_lines.append(line)
                    del ridge_lines[ind]
    
        out_lines = []
        for line in (final_lines + ridge_lines):
            sortargs = numpy.array(numpy.argsort(line[0]))
            rows, cols = numpy.zeros_like(sortargs), numpy.zeros_like(sortargs)
            rows[sortargs] = line[0]
            cols[sortargs] = line[1]
            out_lines.append([rows, cols])
    
        return out_lines

    @staticmethod
    def _filter_ridge_lines(cwt, ridge_lines, min_length=None, min_snr=10, noise_perc=95):
        """
        Filter ridge lines according to prescribed criteria. Intended
        to be used for finding relative maxima.
        
        Parameters
        ----------
        cwt : 2-D ndarray
            Continuous wavelet transform from which the `ridge_lines` were defined.
        ridge_lines : 1-D sequence
            Each element should contain 2 sequences, the rows and columns
            of the ridge line (respectively).
        min_length : int, optional
            Minimum length a ridge line needs to be acceptable.
            Default is ``cwt.shape[0] / 4``, ie 1/4-th the number of widths.
        min_snr : float, optional
            Minimum SNR ratio. Default 10. The signal is the value of
            the cwt matrix at the shortest length scale (``cwt[0, loc]``), the
            noise is the `noise_perc`th percentile of datapoints.
        noise_perc : float, optional
            When calculating the noise floor, percentile of data points
            examined below which to consider noise. Calculated using
            scipy.stats.scoreatpercentile.
        """
        if min_length is None:
            min_length = numpy.ceil(cwt.shape[0] / 4)
    
        def filt_func(line):
            line_id = __name__ + ': '
            line_id += 'ridge line at ({0:d}, {1:d}) '.format(line[0][0], line[1][0])
            if len(line[0]) < min_length:
                logging.debug(line_id + 'failed length test, {0} < {1}.'.
                              format(len(line[0]), min_length))
                return False
            snr = PeakFind._ridge_line_snr(cwt, line, noise_perc)
            if snr < min_snr:
                logging.debug(line_id + 'failed snr test, {0} < {1}.'.
                              format(snr, min_snr))
                return False
            return True
        return list(filter(filt_func, ridge_lines))
    
    @staticmethod
    def _ridge_line_snr(cwt, line, noise_perc=95):
        '''
        Return the signal-to-noise ratio of a ridge line.
        Calculates a negative SNR for negative peaks.

        Parameters
        ----------
        cwt : 2-D ndarray
            Continuous wavelet transform from which the `ridge_lines` were defined.
        line : (1-D sequence, 1-D sequence)
            The rows and columns of the ridge line.
        noise_perc : float, optional
            When calculating the noise floor, percentile of data points
            examined below which to consider noise. Calculated using
            scipy.stats.scoreatpercentile.

        '''
        #print zip(line[0], line[1])
        noise = scipy.stats.mstats.scoreatpercentile(numpy.abs(cwt[0,:]), per=noise_perc)
        signal = cwt[line[0], line[1]]
        maxidxs = PeakFind._argrelmax(signal)
        if len(maxidxs) == 0:
            # ridge lines that do not have a local maximum in cwt vs. width 
            # have a width outside the window specified by widths
            return 0
        else:
            return signal[maxidxs[0]]/noise
    
    @staticmethod
    def _ridge_line_width(cwt, line, widths):
        '''
        Return best guess for the width of a peak by examining its ridge line.
        
        Parameters
        ----------
        cwt : 2-D ndarray
            Continuous wavelet transform from which the `ridge_lines` were defined.
        line : (1-D sequence, 1-D sequence)
            The rows and columns of the ridge line.
        widths : 1-D sequence
            The peak width associated with each row of the cwt matrix
        '''
        signal = cwt[line[0], line[1]]
        maxidxs = PeakFind._argrelmax(signal)
        # fall back to argmax if no relative maximum is present
        if len(maxidxs) == 0:
            # this line should have been filtered out, so this should never happen
            maxidx = numpy.argmax(signal)
        else:
            maxidx = maxidxs[0]
        return widths[line[0][maxidx]]
    
    @staticmethod
    def _argrelmax(xs):
        '''
        Return the indices of relative maxima in vector xs.
        Local maxima at the edges are ignored.
        '''
        boolmax = numpy.all((xs[1:-1]>xs[:-2], xs[1:-1]>xs[2:]), axis=0)
        return [1+i for i in numpy.nonzero(boolmax)[0]]
