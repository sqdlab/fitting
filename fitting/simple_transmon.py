import numpy
from collections import OrderedDict
from . import FitBase, apply_along_axis

class SimpleTransmon(FitBase):
        PARAMETERS = ['Ejmax', 'Ec', 'phi0', 'flux_offset']
    
        @staticmethod
        def f(f, Ejmax, Ec, phi0, flux_offset):
            '''
            f = sqrt(8*Ejmax * abs(cos(pi*(flux_offset+f)/phi0))*Ec) - Ec
            '''
            return numpy.sqrt(8* Ejmax * numpy.abs(numpy.cos(numpy.pi*(flux_offset+f)/phi0)) * Ec) - Ec
        
        # guesser is not implemented. pass guesses via the guess arguments to __init__ or fit
        
class DispersiveTransmon(FitBase):
        PARAMETERS = ['Ejmax', 'Ec', 'phi0', 'flux_offset', 'g', 'fr', 'N']
    
        @staticmethod
        def f(f, Ejmax, Ec, phi0, flux_offset, g, fr, N):
            '''
            fq = sqrt(8*Ejmax * abs(cos(pi*(flux_offset+f)/phi0))*Ec) - Ec
            f = fq - sqrt(N+1)*g**2/(fr-fq)
            '''
            fq = numpy.sqrt(8* Ejmax * numpy.abs(numpy.cos(numpy.pi*(flux_offset+f)/phi0)) * Ec) - Ec
            return fq - numpy.sqrt(N+1)*g**2/(fr-fq)

class DispersiveCavity(FitBase):
    PARAMETERS = ['f_r', 'g']
    
    @staticmethod
    def f(f_q, f_r, g):
        ''' f_r + g**2 / (f_r - f_q) '''
        return f_r + g**2 / (f_r - f_q)