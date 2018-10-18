#*****************************************************************************
#
#  Project:  Hillshade/Aerial Imagery Multiplier
#  Purpose:  Multiplies a hillshade with a defined opacity into an aerial
#            image, simulating a multiply blend mode. Assumes image and
#            hillshade have same extent, cell sizes.
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

from osgeo import gdal
from skimage import exposure
#import numpy as np
import datetime


# Currently assumes that the image and hillshade have same projection, extent,
# and cell size. To accomplish this, assuming your hs extent is larger than the
# imagery's extent, create VRTs of your hillshade with the same resolution and
# extent as the imagery, then use the VRTs as the source hillshade.
# TODO:
# Add raster cell translation to handle heterogeneous cell sizes and extents


def clahe():
    s_fh = gdal.Open(source_dem, gdal.GA_ReadOnly)
    rows = s_fh.RasterYSize
    cols = s_fh.RasterXSize
    driver = gdal.GetDriverByName("GTIFF")
    s_band = s_fh.GetRasterBand(1)
    s_nodata = s_band.GetNoDataValue()

    # Get source georeference info
    transform = s_fh.GetGeoTransform()
    projection = s_fh.GetProjection()
    cell_size = abs(transform[5])  # Assumes square pixels where height=width
    #s_nodata = s_band.GetNoDataValue()

    # get the data in and do somethign noticable to it
    data1 = s_fh.GetRasterBand(1).ReadAsArray()
    #data2 = s_fh.GetRasterBand(2).ReadAsArray()
    #data3 = s_fh.GetRasterBand(3).ReadAsArray()
    #data4 = s_fh.GetRasterBand(4).ReadAsArray()

    #hs_stretch = exposure.equalize_adapthist(data4, 100, .01)
    hs_stretch = exposure.equalize_adapthist(data1, 25, .05)
    # CLAHE Notes:
    #   kernel size: changes what data is emphasized (low number only darkens abrupt changes, like mountains; high number darkens more moderate slopes)
    #   clip limit: how much the data is emphasized (values ~.1 and greater are very contrasty- cool look, but not very usable)

    #hs_mult = (data4 / 255.)
    hs_mult = hs_stretch

    mod1 = data1 * hs_mult
    # mod2 = data2 * hs_mult
    # mod3 = data3 * hs_mult

    hs_stretch *= 255.0/hs_stretch.max()

    # Set up output file
    co = ["compress=jpeg", "tiled=yes", "interleave=pixel", "photometric=ycbcr"]
    lzwco = ["compress=lzw", "tiled=yes"]
    t_fh = driver.Create(out_dem, cols, rows, 1, gdal.GDT_Byte, options=lzwco)
    t_fh.SetProjection(projection)
    t_fh.SetGeoTransform(transform)

    t1 = t_fh.GetRasterBand(1)
    t1.WriteArray(hs_stretch)
    t1.SetNoDataValue(s_nodata)

    # t2 = t_fh.GetRasterBand(2)
    # t2.WriteArray(mod2)
    #
    # t3 = t_fh.GetRasterBand(3)
    # t3.WriteArray(mod3)

    t1 = None
    # t2 = None
    # t3 = None

    t_fh = None


def multiplyhs(image, hs, out, chunk_size, opacity):

    start_time = datetime.datetime.now()
    i_fh = gdal.Open(image, gdal.GA_ReadOnly)
    rows = i_fh.RasterYSize
    cols = i_fh.RasterXSize
    driver = gdal.GetDriverByName("GTIFF")
    # s_band = i_fh.GetRasterBand(1)
    # s_nodata = s_band.GetNoDataValue()

    hs_fh = gdal.Open(hs, gdal.GA_ReadOnly)

    # Get source georeference info
    transform = i_fh.GetGeoTransform()
    projection = i_fh.GetProjection()
    cell_size = abs(transform[5])  # Assumes square pixels where height=width
    #s_nodata = s_band.GetNoDataValue()

    # Set up output file
    jpeg = ["compress=jpeg", "tiled=yes", "interleave=pixel", "photometric=ycbcr", "bigtiff=yes"]
    # lzwco = ["compress=lzw", "tiled=yes"]
    t_fh = driver.Create(out, cols, rows, 3, gdal.GDT_Byte, options=jpeg)
    t_fh.SetProjection(projection)
    t_fh.SetGeoTransform(transform)

    # calculate breaks every chunk_size pixels
    row_splits = list(range(0, rows, chunk_size))
    col_splits = list(range(0, cols, chunk_size))

    # add total number of rows/cols to be last break (used for x/y_end)
    row_splits.append(rows)
    col_splits.append(cols)

    total_chunks = (len(row_splits) - 1) * (len(col_splits) - 1)
    progress = 0

    # Rows = i = y values, cols = j = x values
    for i in range(0, len(row_splits) - 1):
        for j in range(0, len(col_splits) - 1):
            progress += 1
            tile = "{}-{}".format(i, j)
            percent = (progress / total_chunks) * 100
            elapsed = datetime.datetime.now() - start_time
            print("Tile {0}: {1:d} of {2:d} ({3:0.3f}%) started at {4}".format(tile,
                                                        progress, total_chunks,
                                                        percent, elapsed))

            read_x_off = col_splits[j]
            read_y_off = row_splits[i]
            read_x_size = col_splits[j + 1] - read_x_off
            read_y_size = row_splits[i + 1] - read_y_off

            print("xoff: {}, xsize: {}, yoff:{}, ysize:{}".format(read_x_off, read_x_size, read_y_off, read_y_size))
            # get the data in and do somethign noticable to it
            red = i_fh.GetRasterBand(1).ReadAsArray(read_x_off, read_y_off,
                                                    read_x_size, read_y_size)
            green = i_fh.GetRasterBand(2).ReadAsArray(read_x_off, read_y_off,
                                                      read_x_size, read_y_size)
            blue = i_fh.GetRasterBand(3).ReadAsArray(read_x_off, read_y_off,
                                                     read_x_size, read_y_size)

            hs = hs_fh.GetRasterBand(1).ReadAsArray(read_x_off, read_y_off,
                                                    read_x_size, read_y_size)

            # Calculate hs multiple

            # To apply transparency to hillshade, use equation:
            # result = opacity * hs + (1-opacity)*background
            # our background is white, thus background = 255.
            hs_trans = opacity * hs + (1. - opacity) * 255.
            hs_mult = hs_trans / 255.

            #hs_mult = hs / 255.
            newred = red * hs_mult
            newgreen = green * hs_mult
            newblue = blue * hs_mult

            t1 = t_fh.GetRasterBand(1)
            t1.WriteArray(newred, read_x_off, read_y_off)

            t2 = t_fh.GetRasterBand(2)
            t2.WriteArray(newgreen, read_x_off, read_y_off)

            t3 = t_fh.GetRasterBand(3)
            t3.WriteArray(newblue, read_x_off, read_y_off)

    t1 = None
    t2 = None
    t3 = None

    t_fh = None
    i_fh = None
    hs_fh = None


# source_aerial = r'e:\lidar\canyons\dem\multiply\imagevrt\tiled_aerial.vrt'
# source_hs = r'e:\lidar\canyons\dem\multiply\imagevrt\full_hs.vrt'
# dest = r'f:\2018\hs_multiplied_test.tif'

# source_aerial = r'e:\lidar\canyons\dem\multiply\imagevrt\aerial.tif'
# source_hs = r'e:\lidar\canyons\dem\multiply\imagevrt\hs.tif'
# dest = r'e:\lidar\canyons\dem\multiply\imagevrt\transparency_test.tif'

source_aerial = r'e:\a_imagery\1981\1981_merged_partial_halfm.tif'
source_hs = r'e:\Lidar\dsm\merged_dsm.vrt'
dest = r'e:\a_imagery\1981\1981_merged_partial_halfm_mult-dsm.tif'

chunksize = 8096
alpha = .9

multiplyhs(source_aerial, source_hs, dest, chunksize, alpha)

# for i in range(0, 6):
#     source_aerial = r'W:\Aerial Imagery\Area Wide Mosaics 2018\2018-3Inch-2016-9Inch-Combined\{}.tif'.format(i)
#     source_hs = r'f:\2018\hs{}.vrt'.format(i)
#     dest = r'f:\2018\tiled\{}.tif'.format(i)
#     multiplyhs(source_aerial, source_hs, dest, chunksize, alpha)
