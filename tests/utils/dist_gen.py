import math
from re import L
import struct
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm 
import seaborn as sns
import scipy.stats

from numpy import random

def normal_distribution(size, exp, var, lim):
  assert(2 * exp < lim)
  standard_var = math.sqrt(var)
  data = np.array(random.normal(loc= exp, scale = standard_var, size=size)).astype(np.int32)
  for i in range(data.shape[0]):
    if data[i] < 0:
      data[i] = 0
    if data[i] > lim:
      data[i] = lim 
  return data

def uniform_distribution(size, low, high):
    
  assert(low >=0 & low <= high)
  data = random.uniform(low = low, high = high, size = size).astype(np.int32)
  return data


def poisson_distribution(size, exp, lim):
  data = np.array(random.poisson(lam = exp, size = size)).astype(np.int32)
  for i in range(data.shape[0]):
    if data[i] < 0:
      data[i] = 0
    if data[i] > lim:
      data[i] = lim 
  return data

def exponential_distribution(size, exp, lim):
  data = np.array(random.exponential(scale = exp, size = size)).astype(np.int32)
  for i in range(data.shape[0]):
    if data[i] < 0:
      data[i] = 0
    if data[i] > lim:
      data[i] = lim 
  return data





class DataAnalysis:
    """
    Generate a histogram, cdf of the data distribution,
    Get the max, min, mean, and variance of the data.

    Attributes:
        prefix_path: result folder prefix.
    """
    def __init__(self, result_prefix_path):
        self.prefix_path = result_prefix_path
        pass

    def plot_hist_and_kde(self, fname, x):
        plt.xlim((x.min(), x.max()))
        sns.distplot(x, hist=True)
        plt.title(fname + "hist and kde")
        plt.xlabel('result length of range search')
        plt.savefig(self.prefix_path + fname)
        plt.close()
        return 

    def plot_kde(self, fname, x_list, label_list):
        assert(len(x_list) == len(label_list))
        for x, label in zip(x_list, label_list):
            plt.xlim((x.min(), x.max()))
            sns.distplot(x, hist=False, label = label)
        plt.xlabel('result length of range search')
        plt.savefig(self.prefix_path + fname+"_kde")
        plt.close()
        return 

    def plot_ecdf(self, fname, x_list, label_list):
        assert(len(x_list) == len(label_list))
        for x, label in zip(x_list, label_list):
            plt.xlim(x.min(), x.max())
            sns.ecdfplot(data = x, label = label)
        plt.xlabel('result length of range search')
        plt.grid()
        plt.savefig(self.prefix_path + fname+"_cdf")
        plt.close()
        return 
    def save_dis_to_bin_file(self, data, filename):
      with open(self.prefix_path + filename, "wb") as f:
        f.write(struct.pack('I', len(data)))
        f.write(struct.pack('I'*len(data), *data))
        f.close()

def print_mean_var(data):
  print("data len: %d mean: %d var: %d max %d min %d\n" %(len(data), data.mean(), data.var(), data.max(), data.min()))

def save_res(data, aly, file_name):
  print(file_name + ": ")
  print_mean_var(data=data)
  aly.plot_kde(file_name, [data], [file_name])
  aly.plot_ecdf(file_name, [data], [file_name])
  aly.save_dis_to_bin_file(data, filename=file_name)

npts = 100000
mean_k = 1000
max_k = 50000
aly = DataAnalysis("./dis/")

data = normal_distribution(npts, mean_k, 100000, max_k)
save_res(data, aly, "norm")

data = uniform_distribution(npts, 0, 2000)
save_res(data, aly, "uniform")

data = poisson_distribution(npts, mean_k, max_k)
save_res(data, aly, "poisson")

data = exponential_distribution(npts, mean_k, max_k)
save_res(data, aly, "exponential")




