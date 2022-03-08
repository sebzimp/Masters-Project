# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 11:30:22 2022

@author: sebzi
"""
#__spec__ = None

import multiprocessing as mp
from datetime import datetime
import time
start = time.perf_counter()


def my_func(a):
  x = a[0]
  y = a[1]
  print(x+y)

#a =[4,2,3]
#print(mp.cpu_count())
def main():
  pool = mp.Pool(mp.cpu_count())
  a = [[1,2],[3,4]]
  result = pool.map(my_func, a)


if __name__ == "__main__":
  main()
  end = time.perf_counter()
  print(end - start)

