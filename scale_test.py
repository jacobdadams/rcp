import numpy as np
#import datetime
import os
#import subprocess
#import contextlib
#import tempfile
import warnings
#import csv
#import argparse
#import traceback
import math
#import multiprocessing as mp
from astropy.convolution import convolve_fft
#from skimage import exposure
from osgeo import gdal


def scale_factor(val, start, max, z, z_min):
    return ((val - start) / (max - start)) * (z_min - z) + z


in_dem_path = r'c:\gis\data\elevation\slco\scale\stretch.tif'
out_dem_path = r'c:\gis\data\elevation\slco\scale\stretch_out_21_1600_g500d100.tif'
start_elev = 1600
#max_elev = 12000
z = 2
min_z = 1
radius = 500
sigma = 100


# Get source file metadata (dimensions, driver, proj, cell size, nodata)
print("Processing {0:s}...".format(in_dem_path))
s_fh = gdal.Open(in_dem_path, gdal.GA_ReadOnly)
rows = s_fh.RasterYSize
cols = s_fh.RasterXSize
driver = s_fh.GetDriver()
bands = s_fh.RasterCount
s_band = s_fh.GetRasterBand(1)

# Get source georeference info
transform = s_fh.GetGeoTransform()
projection = s_fh.GetProjection()
cell_size = abs(transform[5])  # Assumes square pixels where height=width
s_nodata = s_band.GetNoDataValue()

# [min, max, mean, std_dev]
stats = s_band.GetStatistics(True, True)
print(stats)

# set the max elev based on the dataset
#start_elev = stats[0]
max_elev = stats[1]

in_array = s_band.ReadAsArray()

# scale factor:
# (real - start_elev) / (max_elev - start elev) * (z_min - z) + min_z



# Create new array with s_nodata values set to np.nan (for edges of raster)
nan_array = np.where(in_array == s_nodata, np.nan, in_array)

# build kernel (Gaussian blur function)
# g is a 2d gaussian distribution of size (2*size) + 1
x, y = np.mgrid[-radius:radius + 1, -radius:radius + 1]
# Gaussian distribution
twosig = 2 * sigma**2
g = np.exp(-(x**2 / twosig + y**2 / twosig)) / (twosig * math.pi)

# Convolve the data and Gaussian function (do the Gaussian blur)
# Supressing runtime warnings due to NaNs (they just get hidden by NoData
# masks in the supper_array rebuild anyways)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    # Use the astropy function because fftconvolve does not like np.nan
    #smoothed = fftconvolve(padded_array, g, mode="valid")
    smoothed = convolve_fft(nan_array, g, nan_treatment='interpolate', normalize_kernel=False)
    # Uncomment the following line for a high-pass filter
    #smoothed = nan_array - smoothed



# nan_array = np.where(in_array == s_nodata, np.nan, in_array)
# diameter = 2 * radius + 1
# # Create a circular mask
# y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
# mask = x**2 + y**2 > radius**2
# # Determine number of Falses (ie, cells in kernel not masked out)
# valid_entries = mask.size - np.count_nonzero(mask)
# # Create a kernel of 1/(the number of valid entries after masking)
# kernel = np.ones((diameter, diameter)) / (valid_entries)
# # Mask away the non-circular areas
# kernel[mask] = 0
#
# # kernel = [[4.5, 0, 0],
# #           [0, 0.001, 0],
# #           [0, 0, -5]]
#
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore", category=RuntimeWarning)
#     smoothed = convolve_fft(nan_array, kernel,
#                                  nan_treatment='interpolate')#, normalize_kernel=False)




# map scale_factor into smoothed array, then multiply nan_array by scale_array
scale_array = scale_factor(smoothed, start_elev, max_elev, z, min_z)

# Where the source array is below our scale area, val = elev * z, else NaN
mask_arrray = np.where(in_array < start_elev, in_array*z, np.nan)

scale_mult = np.multiply(nan_array, scale_array)
z_mult = np.multiply(in_array, z)
scaled = np.where(in_array < start_elev, z_mult, scale_mult)
#scaled = np.multiply(nan_array, scale_array)

print("scale_array mean: {}".format(np.mean(scale_array)))
print("scaled mean: {}".format(np.mean(scaled)))
print("max: {}\nmin: {}".format(max_elev, start_elev))

del s_band
del s_fh

dtype = gdal.GDT_Float32
lzw_opts = ["compress=lzw", "tiled=yes", "bigtiff=yes"]
t_fh = driver.Create(out_dem_path, cols, rows, bands, dtype, options=lzw_opts)
t_fh.SetGeoTransform(transform)
t_fh.SetProjection(projection)
t_band = t_fh.GetRasterBand(1)
t_band.SetNoDataValue(s_nodata)
t_band.WriteArray(scaled)

del t_band
del t_fh
