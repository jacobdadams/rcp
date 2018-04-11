#*****************************************************************************
#
#  Project:  GDAL Skyshade
#  Purpose:  Create "skyshade" hillshades using GDAL using the method described
#            in Kennelly and Stewart (2014), General sky models for
#            illuminating terrains.
#  Author:   Jacob Adams, jacob.adams@cachecounty.org
#
#*****************************************************************************
# MIT License
#
# Copyright (c) 2018 Cache County
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#*****************************************************************************

#*****************************************************************************
# NOTE
# This script will be updated and incorporated into raster_chunk_processing.py
#
#*****************************************************************************

from __future__ import print_function
from osgeo import gdal
import numpy as np
import csv
import datetime
import os

def hillshade(array, az, alt, p_height, p_width, scale):
    '''
    Hillshade algorithm based on comments in the gdal DEMProcessing routines.
    '''

    x = np.zeros(array.shape)
    y = np.zeros(array.shape)
    az = 360.-az
    azrad = az * np.pi/180.
    altrad = alt * np.pi/180.

    x, y = np.gradient(array, p_width, p_height, edge_order=2)

    # sinalt = np.sin(altrad)
    # cosaz = np.cos(azrad)
    # cosalt = np.cos(altrad)
    # sinaz = np.sin(azrad)
    # xx_plus_yy = x*x + y*y
    # shaded = (sinalt - (y * cosaz * cosalt - x * sinaz * cosalt)) / np.sqrt(1+xx_plus_yy)

    shaded = (np.sin(altrad) - (y * np.cos(azrad) * np.cos(altrad) - x * np.sin(azrad) * np.cos(altrad))) / np.sqrt(1+(x*x + y*y))

    return shaded*255

def skyshade(in_dem, lum_file, out_path, temp_dir, ex_factor, size_limit):
    '''
    Implements sky model as described in Kennelly and Stewart (2014), General
    sky models for illuminating terrains.
    in_dem:         Path to input DEM
    lum_file:       Path to csv file containing az/alt/weight lines
    out_path:       Path for completed file
    temp_dir:       Path to directory that will hold the pickled chunks
    ex_factor:      Exageration factor
    size_limit:     Chunk size
    '''

    # Load csv into lines array
    # break DEM into chunks, pickle chunks
    # for each chunk:
    #   for each line:
    #       calculate hillshade
    #       multiply resulting hillshade by factor
    #       add hillshade to existing running sum
    #   save chunk
    # Recombine chunks, save as output

    pickle_dir = temp_dir

    gdal.UseExceptions()

    start = datetime.datetime.now()
    print("Started: " + start.strftime("%Y-%m-%d %H:%M:%S"))

    ds = gdal.Open(in_dem, gdal.GA_ReadOnly)
    rows = ds.RasterYSize
    cols = ds.RasterXSize
    driver = ds.GetDriver()

    # get georeference info
    transform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]

    # read in data (only one band in dem) and store as array
    elev_data = ds.ReadAsArray()

    elev_data = elev_data * ex_factor

    # Read in luminance file, loop line by line
    print("Reading in luminance file %s" %(lum_file))
    lines = []
    with open(lum_file, 'r') as l:
        reader = csv.reader(l)
        for line in reader:
            lines.append(line)

    # Split dem into chunks for processing
    print("Splitting DEM into chunks of %d by %d pixels" %(size_limit, size_limit))
    sub_arrays = {}
    keys = []
    if rows > size_limit and cols > size_limit:
        #Split it

        # calculate breaks every size_limit pixels
        row_splits = range(0, rows, size_limit)
        col_splits = range(0, cols, size_limit)

        # add total number of rows/cols to be last break
        row_splits.append(rows)
        col_splits.append(cols)

        # Split into 2000x2000 pixel chunks
        for i in range(len(row_splits)):
            if i < len(row_splits)-1:
                for j in range(len(col_splits)):
                    if j < len(col_splits)-1:
                        key = "%d;%d;%d;%d" % (row_splits[i], row_splits[i+1], col_splits[j], col_splits[j+1])
                        # gives us [0:2,0:2], [0:2,2:4]... [4:5,4:5] where [rows, cols]
                        keys.append(key)
                        data = elev_data[row_splits[i]:row_splits[i+1],col_splits[j]:col_splits[j+1]]
                        # make our new nodata value -999.
                        for k in np.nditer(data, op_flags=['readwrite']):
                            if k < 0:
                                k[...] = -999.
                        np.save(os.path.join(pickle_dir, key), data)

    else:
        key = "0;%d;0;%d" % (rows, cols)
        keys.append(key)
        np.save(os.path.join(pickle_dir, key), elev_data)

    # Memory managment
    elev_data = None

    # Calculate skyshade for each subarray
    print("Looping through chunks")
    skyshades = []
    counter = 0
    for key in keys:
        counter = counter + 1
        print("Chunk %s, %d of %d" %(key, counter, len(keys)))
        pickle_path = os.path.join(pickle_dir, key) + ".npy"
        chunk = np.load(pickle_path)

        # initialize total skyshade for this chunk as 0's
        skyshade_chunk = np.zeros((chunk.shape))

        # Loop through luminance file lines to calculate multiple hillshades for that chunk
        for line in lines:
            az = float(line[0])
            alt = float(line[1])
            weight = float(line[2])

            # Modify azimuth to proper value for maths
            az = 180. - az
            if az < 0.:
                az = az + 360.

            tfile_source = "c:\\temp\\gis\\skyshade\\temp\\tempsource.tif"
            tfile_dest = "c:\\temp\\gis\\skyshade\\temp\\tempdest.tif"
            # Must specify gdal.GDT_Float32 as the data type here, otherwise it will default to 8-bit and lose all information
            temp_ds = driver.Create(tfile_source, chunk.shape[1], chunk.shape[0], 1, gdal.GDT_Float32)
            temp_band = temp_ds.GetRasterBand(1)
            temp_band.SetNoDataValue(-999.)
            temp_band.WriteArray(chunk)
            temp_band = None
            temp_ds.SetProjection(projection)

            shade = gdal.DEMProcessing(tfile_dest, temp_ds, "hillshade", azimuth = az, altitude = alt, computeEdges = True, scale = pixelWidth).ReadAsArray() * weight#.astype(np.uint8)

            temp_ds = None

            # calculate a new hillshade array, multiplied by weight
            #shade = hillshade(chunk, az, alt, pixelHeight, pixelWidth, 1) * weight

            # multiply this array by the weight
            # hs2 = hs1 * weight

            # sum the new hillshade array with the sum of the previous hillshades array
            #total = previous + shade

            # copy (deep copy? reference? may cause headaches...) for next iteration
            #previous = total

            skyshade_chunk = skyshade_chunk + shade
            print('.', end='')

        # new keys (filenames for completed chunks) have ';99999' appended; appended part must be an int to work with the map(int, ...) below.
        shadedkey = key + ";99999"
        skyshades.append(shadedkey)

        np.save(os.path.join(pickle_dir, shadedkey), skyshade_chunk)
        print("")

    #Rebuild into one large array
    print("Rebuilding chunks into %s" %(out_path))
    final = np.zeros((rows, cols))#.astype(Byte)
    for part in skyshades:
        sky_path = os.path.join(pickle_dir, part) + ".npy"
        skyshade = np.load(sky_path)
        s = map(int, part.split(';'))
        final[s[0]:s[1],s[2]:s[3]] = skyshade

    # Write out array as final skyshade
    dest = driver.Create(out_path, cols, rows, 1, gdal.GDT_Float32)
    band = dest.GetRasterBand(1)
    band.WriteArray(final)
    dest.SetGeoTransform(transform)
    dest.SetProjection(projection)

    # Delete objects, causing destination data set to write to disk. Must delete
    # band before dataset to prevent gdal gotcha
    band = None
    dest = None

    finish = datetime.datetime.now() - start
    print("Total time: " + str(finish))



dem = "c:\\temp\\gis\\dem_state.tif"

# Exageration factor;
ex = 1
size = 2000

out_dir = "c:\\temp\\gis\\skyshade\\skyshades"
temp_dir = "c:\\temp\\gis\\skyshade\\temp"
lum = "c:\\temp\\gis\\skyshade\\lum\\1_45_315_150.csv"
out = out_dir + "\\1_45_315_150.tif"

skyshade(dem, lum, out, temp_dir, ex, size)

skyshade(dem, "c:\\temp\\gis\\skyshade\\lum\\2_45_315_150.csv", out_dir + "\\2_45_315_150.tif", temp_dir, ex, size)
skyshade(dem, "c:\\temp\\gis\\skyshade\\lum\\3_45_315_150.csv", out_dir + "\\3_45_315_150.tif", temp_dir, ex, size)
skyshade(dem, "c:\\temp\\gis\\skyshade\\lum\\5_45_315_150.csv", out_dir + "\\5_45_315_150.tif", temp_dir, ex, size)

# for i in range (6,10):
    # filename = str(i) + "_45_215.csv"
    # lum = "c:\\temp\\gis\\skyshade\\lum\\" + filename
    # out = "c:\\temp\\gis\\skyshade\\out\\" + str(i) + "_45_215.tif"
    # skyshade(dem, lum, out, ex, size)

# for path, x, name in os.walk("C:\\temp\\gis\\skyshade\\lum"):
    # for fname in name:
        # if fname.partition('.')[2] == 'csv':
            # lum = os.path.join(path, fname)
            # no_csv = fname.partition('.')[0]
            # tif = no_csv + ".tif"
            # out = os.path.join(out_dir, tif)
            # print("Running: " + no_csv)
            # skyshade(dem, lum, out, temp_dir, ex, size)
