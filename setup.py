from setuptools import setup
import time

setup(
    name='fitting',
    version=time.strftime('%Y%m%d'),
    author='Rohit Navarathna, Markus Jerger',
    author_email='r.navarathna@uq.edu.au',
    description='A python toolbox for fitting curves.',
    #long_description=''
    #license='',
    keywords='fitting, curve fitting',
    url='http://sqd.equs.org/',
    packages=['fitting'], 
    zip_safe=False,
    python_requires='''
        >=3.0
    ''', 
    install_requires='''
        uqtools
    '''
)