'''
MODEL    :: Tensorflow Computer Vision Platform
DATE    :: 2020-01-23
FILE     :: tfcv_main.py 
'''

from __future__ import absolute_import, division, print_function

from config.option import Options
from engine.train import Train
from engine.test import Test
from engine.single import Single

options = Options()
params = options.parse()


if __name__ == "__main__":
    if params.mode == "train":
        trainer = Train(params)
        trainer.train()
    if params.mode == "test":
        tester = Test(params)
        tester.test()
    if params.mode == "demo":
        tester = Test(params)
        tester.test()    
    if params.mode == "single":
        tester = Single(params)
        tester.single()                
    else:
        print("--- Wrong Mode ---")
