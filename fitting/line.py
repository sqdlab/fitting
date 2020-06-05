from . import FitBase

class Line(FitBase):
        PARAMETERS = ['m', 'c']
    
        @staticmethod
        def f(x, m, c):
            '''
            m*x + c
            '''
            return m*x+c