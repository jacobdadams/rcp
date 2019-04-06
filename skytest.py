from osgeo import gdal
import numpy as np
import os
import warnings
import csv
import math
import datetime
import numba
import sys
from memory_profiler import profile


# Notes:
# Shading is memory intensive (lots of copies of the array), Shadowing is processor intensive.
# Numba jit'ing the monolithic skymodel doesn't speed it up by much- just jit'ing shade is fine.
#   putting it into a single loop slows it down even further- 34s vs 24s for non-monolithic
# The "return shaded*255" line holds onto a lot of memory- put the *255 in the body of the method.

def sizeof_fmt(num, suffix='B'):
    '''
    Quick-and-dirty method for formating file size, from Sridhar Ratnakumar,
    https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size.
    '''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


@numba.jit("f8[:,:](f4[:,:],f8,f8,f8,f8)", nopython=True)
def hillshade_numba(in_array, az, alt, res, nodata):
    '''
    Custom implmentation of hillshading, using the algorithm from the source
    code for gdaldem. The inputs and outputs are the same as in gdal or ArcGIS.
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    az:             The sun's azimuth, in degrees.
    alt:            The sun's altitude, in degrees.
    scale:          When true, stretches the result to 1-255. CAUTION: If using
                    as part of a parallel or multi-chunk process, each chunk
                    has different min and max values, which leads to different
                    stretching for each chunk.
    '''

    # Create new array with s_nodata values set to np.nan (for edges)
    nan_array = np.where(in_array == nodata, np.nan, in_array)

    # Initialize shaded array to 0s
    shaded = np.zeros(nan_array.shape)

    # Conversion between mathematical and nautical azimuth
    az = 90. - az

    azrad = az * np.pi / 180.
    altrad = alt * np.pi / 180.

    sinalt = np.sin(altrad)
    cosaz = np.cos(azrad)
    cosalt = np.cos(altrad)
    sinaz = np.sin(azrad)

    rows = nan_array.shape[0]
    cols = nan_array.shape[1]

    for i in range(1, rows-1):  # ignoring edges right now by subsetting
        for j in range(1, cols-1):
            window = nan_array[i-1:i+2, j-1:j+2].flatten()

            x = ((window[2] + 2. * window[5] + window[8]) -
                 (window[0] + 2. * window[3] + window[6])) / (8. * res)

            y = ((window[6] + 2. * window[7] + window[8]) -
                 (window[0] + 2. * window[1] + window[2])) / (8. * res)

            xx_plus_yy = x * x + y * y
            alpha = y * cosaz * cosalt - x * sinaz * cosalt
            shade = (sinalt - alpha) / np.sqrt(1 + xx_plus_yy)

            shaded[i,j] = shade * 255

    # scale from 0-1 to 0-255
    return shaded


#@profile
def hillshade(in_array, az, alt, res, nodata):
    '''
    Custom implmentation of hillshading, using the algorithm from the source
    code for gdaldem. The inputs and outputs are the same as in gdal or ArcGIS.
    in_array:       The input array, should be read using the supper_array
                    technique from below.
    az:             The sun's azimuth, in degrees.
    alt:            The sun's altitude, in degrees.
    scale:          When true, stretches the result to 1-255. CAUTION: If using
                    as part of a parallel or multi-chunk process, each chunk
                    has different min and max values, which leads to different
                    stretching for each chunk.
    '''

    # Create new array wsith s_nodata values set to np.nan (for edges)
    nan_array = np.where(in_array == nodata, np.nan, in_array)

    # x = np.zeros(nan_array.shape)
    # y = np.zeros(nan_array.shape)

    # Conversion between mathematical and nautical azimuth
    az = 90. - az

    azrad = az * np.pi / 180.
    altrad = alt * np.pi / 180.

    x, y = np.gradient(nan_array, res, res, edge_order=2)

    nan_array = None

    sinalt = np.sin(altrad)
    cosaz = np.cos(azrad)
    cosalt = np.cos(altrad)
    sinaz = np.sin(azrad)
    xx_plus_yy = x * x + y * y
    alpha = y * cosaz * cosalt - x * sinaz * cosalt
    x = None
    y = None
    shaded = (sinalt - alpha) / np.sqrt(1 + xx_plus_yy) * 255
    # print('')
    # print("Locals")
    # for var, obj in locals().items():
    #     print(var, sizeof_fmt(sys.getsizeof(obj)))
    # print("\nGlobals")
    # for var, obj in globals().items():
    #     print(var, sizeof_fmt(sys.getsizeof(obj)))
    # scale from 0-1 to 0-255
    return shaded# * 255


@numba.jit(nopython=True)
def skymodel_numba(in_array, lum_lines, res, nodata):#, cell_size):
    '''
    Creates a unique hillshade based on a skymodel, implmenting the method
    defined in Kennelly and Stewart (2014), A Uniform Sky Illumination Model to
    Enhance Shading of Terrain and Urban Areas.

    in_array:       The input array, should be read using the supper_array
                    technique from below.
    lum_lines:      The azimuth, altitude, and weight for each iteration of the
                    hillshade. Stored as an array lines, with each line being
                    an array of [az, alt, weight].
    '''

    # initialize skyshade as 0's
    skyshade = np.zeros((in_array.shape))

    # If it's all NoData, just return an array of 0's
    if in_array.mean() == nodata:
        return skyshade

    # Multiply by 5 per K & S
    in_array *= 5
    # Loop through luminance file lines to calculate multiple hillshades
    for l in range(0, lum_lines.shape[0]):
    #for line in lum_lines:
        az = lum_lines[l, 0]
        alt = lum_lines[l, 1]
        weight = lum_lines[l, 2]
        # print("shading...")

        # Hillshade variables
        nan_array = np.where(in_array == nodata, np.nan, in_array)

        # Initialize shaded array to 0s
        shaded = np.zeros(nan_array.shape)

        # Conversion between mathematical and nautical azimuth
        az = 90. - az

        azrad = az * np.pi / 180.
        altrad = alt * np.pi / 180.

        sinalt = math.sin(altrad)
        cosaz = math.cos(azrad)
        cosalt = math.cos(altrad)
        sinaz = math.sin(azrad)

        rows = nan_array.shape[0]
        cols = nan_array.shape[1]

        # Shadow variables
        delta_j = math.cos(azrad)
        delta_i = -1. * math.sin(azrad)
        tanaltrad = math.tan(altrad)

        mult_size = 1
        max_steps = 200

        shadow_array = np.ones(in_array.shape)  # init to 1 (not shadowed), change to 0 if shadowed
        max_elev = np.max(in_array)

        # Just loop once, doing both hillshade and shadow
        for i in range(1, rows-1):  # ignoring edges right now by subsetting
            for j in range(1, cols-1):

                # ===================
                # Hillshade algorithm
                window = nan_array[i-1:i+2, j-1:j+2].flatten()

                x = ((window[2] + 2. * window[5] + window[8]) -
                     (window[0] + 2. * window[3] + window[6])) / (8. * res)

                y = ((window[6] + 2. * window[7] + window[8]) -
                     (window[0] + 2. * window[1] + window[2])) / (8. * res)

                xx_plus_yy = x * x + y * y
                alpha = y * cosaz * cosalt - x * sinaz * cosalt
                shade = (sinalt - alpha) / np.sqrt(1 + xx_plus_yy)

                shaded[i,j] = shade * 255

                # ================
                # Shadow algorithm
                point_elev = in_array[i, j]  # the point we want to determine if in shadow
                # start calculating next point from the source point
                prev_i = i
                prev_j = j

                # shadow = 1  # 0 if shadowed, 1 if not

                for p in range(0, max_steps):
                    # Figure out next point along the path
                    next_i = prev_i + delta_i * p * mult_size
                    next_j = prev_j + delta_j * p * mult_size
                    # Update prev_i/j for next go-around
                    prev_i = next_i
                    prev_j = next_j

                    # We need integar indexes for the array
                    idx_i = int(round(next_i))
                    idx_j = int(round(next_j))

                    # distance for elevation check is distance in cells (idx_i/j), not distance along the path
                    # critical height is the elevation that is directly in the path of the sun at given alt/az
                    idx_distance = math.sqrt((i - idx_i)**2 + (j - idx_j)**2)
                    # path_distance = math.sqrt((i - next_i)**2 + (j - next_j)**2)
                    critical_height = idx_distance * tanaltrad * res + point_elev

                    in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
                    in_height = critical_height < max_elev
                    # in_distance = path_distance * res < max_distance

                    if in_bounds and in_height: # and in_distance:
                        next_elev = in_array[idx_i, idx_j]
                        if next_elev > point_elev and next_elev > critical_height:
                            shadow_array[i, j] = 0
                            # print(p)
                            break  # We're done with this point, move on to the next

        # weight the shaded array (mult is cummutative, same as (shadow*shade)*weight)
        shaded *= weight

        # # print("shadowing...")
        # #--- SHADOWS ---
        # shadow_array = np.ones(in_array.shape)  # init to 1 (not shadowed), change to 0 if shadowed
        # max_elev = np.max(in_array)
        # # max_distance = 500.
        #
        # az = 90. - az  # convert from 0 = north, cw to 0 = east, ccw
        #
        # azrad = az * np.pi / 180.
        # altrad = alt * np.pi / 180.
        # delta_j = math.cos(azrad)
        # delta_i = -1. * math.sin(azrad)
        # tanaltrad = math.tan(altrad)
        #
        # mult_size = 1
        # max_steps = 200
        # for i in range(0, rows):
        #     for j in range(0, cols):
        #         point_elev = in_array[i, j]  # the point we want to determine if in shadow
        #         # start calculating next point from the source point
        #         prev_i = i
        #         prev_j = j
        #
        #         # shadow = 1  # 0 if shadowed, 1 if not
        #
        #         for p in range(0, max_steps):
        #             # Figure out next point along the path
        #             next_i = prev_i + delta_i * p * mult_size
        #             next_j = prev_j + delta_j * p * mult_size
        #             # Update prev_i/j for next go-around
        #             prev_i = next_i
        #             prev_j = next_j
        #
        #             # We need integar indexes for the array
        #             idx_i = int(round(next_i))
        #             idx_j = int(round(next_j))
        #
        #             # distance for elevation check is distance in cells (idx_i/j), not distance along the path
        #             # critical height is the elevation that is directly in the path of the sun at given alt/az
        #             idx_distance = math.sqrt((i - idx_i)**2 + (j - idx_j)**2)
        #             # path_distance = math.sqrt((i - next_i)**2 + (j - next_j)**2)
        #             critical_height = idx_distance * tanaltrad * res + point_elev
        #
        #             in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
        #             in_height = critical_height < max_elev
        #             # in_distance = path_distance * res < max_distance
        #
        #             if in_bounds and in_height: # and in_distance:
        #                 next_elev = in_array[idx_i, idx_j]
        #                 if next_elev > point_elev and next_elev > critical_height:
        #                     shadow_array[i, j] = 0
        #                     # print(p)
        #                     break  # We're done with this point, move on to the next

        #shadowed = shadows(in_array, az, alt, res, nodata)
        print("combining...")
        skyshade = skyshade + shaded * shadow_array


    return skyshade


def skymodel(in_array, lum_lines, res, nodata):#, cell_size):
    '''
    Creates a unique hillshade based on a skymodel, implmenting the method
    defined in Kennelly and Stewart (2014), A Uniform Sky Illumination Model to
    Enhance Shading of Terrain and Urban Areas.

    in_array:       The input array, should be read using the supper_array
                    technique from below.
    lum_lines:      The azimuth, altitude, and weight for each iteration of the
                    hillshade. Stored as an array lines, with each line being
                    an array of [az, alt, weight].
    '''

    # initialize skyshade as 0's
    skyshade = np.zeros((in_array.shape), dtype=np.float32)

    # If it's all NoData, just return an array of 0's
    if in_array.mean() == nodata:
        return skyshade

    # Multiply by 5 per K & S
    in_array *= 5
    # Loop through luminance file lines to calculate multiple hillshades
    for line in lum_lines:
        az = line[0]
        alt = line[1]
        weight = line[2]
        print("shading...")
        # shade = hillshade_numba(in_array, az, alt, res, nodata) * weight
        shade = hillshade(in_array[:], az, alt, res, nodata) * weight
        print("shadowing...")
        shadowed = shadows(in_array, az, alt, res, nodata)
        print("combining...")

        skyshade += (shade * shadowed)
        # print(np.nanmean(skyshade))
        # for var, obj in locals().items():
        #     print(var, sizeof_fmt(sys.getsizeof(obj)))
        shade = None
        shadowed = None

    return skyshade


@numba.jit("u1[:,:](f4[:,:],f8,f8,f8,f8)", nopython=True)
def shadows(in_array, az, alt, res, nodata):
    # Rows = i = y values, cols = j = x values
    rows = in_array.shape[0]
    cols = in_array.shape[1]
    shadow_array = np.ones(in_array.shape, dtype=np.uint8)  # init to 1 (not shadowed), change to 0 if shadowed
    max_elev = np.max(in_array)
    # max_distance = 500.

    az = 90. - az  # convert from 0 = north, cw to 0 = east, ccw

    azrad = az * np.pi / 180.
    altrad = alt * np.pi / 180.
    delta_j = math.cos(azrad)  # these are switched sin for cos because of 90-az above
    delta_i = -1. * math.sin(azrad)  # these are switched sin for cos because of 90-az above
    tanaltrad = math.tan(altrad)

    # Mult size is in array units, not georef units
    mult_size = 1
    max_steps = 200

    counter = 0
    max = rows * cols

    # numba jit w/for loop is faster than nditer
    # # https://docs.scipy.org/doc/numpy/reference/arrays.nditer.html
    # it = np.nditer(in_array, flags=['multi_index'])
    # while not it.finished:
    #     i = it.multi_index[0]
    #     j = it.multi_index[1]
    #
    #     point_elev = it[0]  # the point we want to determine if in shadow
    #     # start calculating next point from the source point
    #     prev_i = i
    #     prev_j = j
    #
    #     for p in range(0, max_steps):
    #         # Figure out next point along the path
    #         next_i = prev_i + delta_i * p * mult_size
    #         next_j = prev_j + delta_j * p * mult_size
    #         # Update prev_i/j for next go-around
    #         prev_i = next_i
    #         prev_j = next_j
    #
    #         # We need integar indexes for the array
    #         idx_i = int(round(next_i))
    #         idx_j = int(round(next_j))
    #
    #         # distance for elevation check is distance in cells (idx_i/j), not distance along the path
    #         # critical height is the elevation that is directly in the path of the sun at given alt/az
    #         idx_distance = math.sqrt((i - idx_i)**2 + (j - idx_j)**2)
    #         # path_distance = math.sqrt((i - next_i)**2 + (j - next_j)**2)
    #         critical_height = idx_distance * tanaltrad * res + point_elev
    #
    #         in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
    #         in_height = critical_height < max_elev
    #         # in_distance = path_distance * res < max_distance
    #
    #         if in_bounds and in_height: # and in_distance:
    #             next_elev = in_array[idx_i, idx_j]
    #             if next_elev > point_elev and next_elev > critical_height:
    #                 shadow_array[i, j] = 0
    #                 # print(p)
    #                 break  # We're done with this point, move on to the next
    #     it.iternext()
    # return shadow_array



    for i in range(0, rows):
        for j in range(0, cols):
            # keep_going = True

            # counter += 1
            # elapsed = datetime.datetime.now() - start
            # print("{}, {}   {}".format(i, j, elapsed))

            point_elev = in_array[i, j]  # the point we want to determine if in shadow
            # start calculating next point from the source point
            prev_i = i
            prev_j = j

            # shadow = 1  # 0 if shadowed, 1 if not

            for step in range(1, max_steps):  # start at a step of 1- a point cannot be shadowed by itself
                # Figure out next point along the path
                # use i/j + delta_i/j instead of prev_i/j + delta_i/j because step takes care of progression for us
                next_i = i + delta_i * step * mult_size
                next_j = j + delta_j * step * mult_size

                # We need integar indexes for the array
                idx_i = int(round(next_i))
                idx_j = int(round(next_j))

                # distance for elevation check is distance in cells (idx_i/j), not distance along the path
                # critical height is the elevation that is directly in the path of the sun at given alt/az
                idx_distance = math.sqrt((i - idx_i)**2 + (j - idx_j)**2)
                # path_distance = math.sqrt((i - next_i)**2 + (j - next_j)**2)
                critical_height = idx_distance * tanaltrad * res + point_elev

                in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
                in_height = critical_height < max_elev
                # in_distance = path_distance * res < max_distance
                #print(in_bounds)

                # set to shaded if out of bounds
                # if in_bounds is False:
                #     shadow_array[i,j] = 0
                #     break

                if in_bounds and in_height:  # and in_distance:
                    next_elev = in_array[idx_i, idx_j]
                    if next_elev > point_elev and next_elev > critical_height:
                        shadow_array[i, j] = 0
                        # print(p)
                        break  # We're done with this point, move on to the next

    return shadow_array

            # while keep_going:  # this inner loop loops through the possible values for each path
            #     # Figure out next point along the path
            #     # delta_j = math.cos(azrad) * res
            #     # delta_i = math.sin(azrad) * res
            #     next_i = prev_i + delta_i
            #     next_j = prev_j + delta_j
            #     # Update prev_i/j for next go-around
            #     prev_i = next_i
            #     prev_j = next_j
            #
            #     # We need integar indexes for the array
            #     idx_i = int(round(next_i))
            #     idx_j = int(round(next_j))
            #
            #     shadow = 1  # 0 if shadowed, 1 if not
            #
            #     # distance for elevation check is distance from cell centers (idx_i/j), not distance along the path
            #     # critical height is the elevation that is directly in the path of the sun at given alt/az
            #     idx_distance = math.sqrt((i - idx_i)**2 + (j - idx_j)**2)
            #     path_distance = math.sqrt((i - next_i)**2 + (j - next_j)**2)
            #     critical_height = idx_distance * tanaltrad * res + point_elev
            #
            #
            #     in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
            #     in_height = critical_height < max_elev
            #     in_distance = path_distance * res < max_distance
            #     #print("{}, {}, {}".format(in_bounds, in_height, in_distance))
            #
            #     if in_bounds and in_height and in_distance:
            #     # bounds check (array bounds, elevation check)
            #     # if idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols and critical_height < max_elev:
            #         next_elev = in_array[idx_i, idx_j]
            #         if next_elev > point_elev:  # only check if the next elev is greater than the point elev
            #             if next_elev > critical_height:
            #                 shadow = 0
            #                 keep_going = False  # don't bother continuing to check
            #
            #     else:
            #         keep_going = False  # our next index would be out of bounds, we've reached the edge of the array
            #
            #
            #     #print("i:{}, j:{}; idx_i:{}, idx_j:{}; idx_distance:{}, critical_height:{}, delta_i:{}, delta_j:{}".format(i, j, idx_i, idx_j, idx_distance, critical_height, delta_i, delta_j))
            #
            # shadow_array[i, j] = shadow  # assign shadow value to output array
            # #print("{}, {}: {}".format(i, j, shadow))

    # return shadow_array


# variables
csv_path = r'C:\GIS\Data\Elevation\Uintahs\test10_nohdr.csv'
in_dem_path = r'C:\GIS\Data\Elevation\Uintahs\utest.tif'
# in_dem_path = r'C:\GIS\Data\Elevation\Uintahs\uintahs_fft60_sub.tif'
out_dem_path = r'C:\GIS\Data\Elevation\Uintahs\utest_sky_shadowedgetest_180_25_stepjusti.tif'

alt = 45.
az = 315.

start = datetime.datetime.now()

gdal.UseExceptions()

lines = []

with open(csv_path, 'r') as l:
    reader = csv.reader(l)
    for line in reader:
        lines.append([float(line[0]), float(line[1]), float(line[2])])

nplines = np.zeros((len(lines), 3))
for i in range(0, len(lines)):
    for j in range(0,3):
        nplines[i,j] = lines[i][j]

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

if os.path.exists(out_dem_path):
    raise IOError("Output file {} already exists.".format(out_dem_path))

print("Reading array")
s_data = s_band.ReadAsArray()

print("s_data: {}".format(sizeof_fmt(sys.getsizeof(s_data))))

# Close source file handle
s_band = None
s_fh = None

print("Processing array")

#shade = hillshade_numba(s_data, az, alt, cell_size)
# shadowed = shadows(s_data, az, alt, cell_size)
# mult = shade * shadowed
#sky = skymodel_numba(s_data, nplines, cell_size, s_nodata)
lines = [[180., 25., 1.]]
sky = skymodel(s_data[:], lines, cell_size, s_nodata)
# Test is 225 az, 25 alt
# shad = shadows(s_data, az, alt, cell_size)

out_array = sky

print("Writing output array")
# Set up target file in preparation for future writes
# If we've been given a vrt as a source, force the output to be geotiff
if driver.LongName == 'Virtual Raster':
    driver = gdal.GetDriverByName('gtiff')
# if os.path.exists(out_dem_path):
#     raise IOError("Output file {} already exists.".format(out_dem_path))

lzw_opts = ["compress=lzw", "tiled=yes", "bigtiff=yes"]

t_fh = driver.Create(out_dem_path, cols, rows, bands, gdal.GDT_Float32, options=lzw_opts)
t_fh.SetGeoTransform(transform)
t_fh.SetProjection(projection)
t_band = t_fh.GetRasterBand(1)
if bands == 1:
    t_band.SetNoDataValue(s_nodata)

t_band.WriteArray(out_array)

t_band = None
t_fh = None

end = datetime.datetime.now()
print(end - start)
