from osgeo import gdal
import numpy as np
import os
import warnings
import csv
import math


def hillshade(in_array, az, alt, scale=False):
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
    nan_array = np.where(in_array == s_nodata, np.nan, in_array)

    x = np.zeros(nan_array.shape)
    y = np.zeros(nan_array.shape)

    # Conversion between mathematical and nautical azimuth
    az = 90. - az

    azrad = az * np.pi / 180.
    altrad = alt * np.pi / 180.

    x, y = np.gradient(nan_array, cell_size, cell_size, edge_order=2)

    sinalt = np.sin(altrad)
    cosaz = np.cos(azrad)
    cosalt = np.cos(altrad)
    sinaz = np.sin(azrad)
    xx_plus_yy = x * x + y * y
    alpha = y * cosaz * cosalt - x * sinaz * cosalt
    shaded = (sinalt - alpha) / np.sqrt(1 + xx_plus_yy)

    # scale from 0-1 to 0-255
    shaded255 = shaded * 255

    if scale:
        # Scale to 1-255 (stretches min value to 1, max to 255)
        # ((newmax-newmin)(val-oldmin))/(oldmax-oldmin)+newmin
        # Supressing runtime warnings due to NaNs (they just get hidden by
        # NoData masks in the supper_array rebuild anyways)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            newmax = 255
            newmin = 1
            oldmax = np.nanmax(shaded255)
            oldmin = np.nanmin(shaded255)

        result = (newmax-newmin) * (shaded255-oldmin) / (oldmax-oldmin) + newmin
    else:
        result = shaded255

    return result


def skymodel(in_array, lum_lines):
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
    if in_array.mean() == s_nodata:
        return skyshade

    # Loop through luminance file lines to calculate multiple hillshades
    for line in lum_lines:
        az = float(line[0])
        alt = float(line[1])
        weight = float(line[2])
        # print("{}, {}, {}".format(az, alt, weight))
        shade = hillshade(in_array, az=az, alt=alt, scale=False) * weight
        # print(np.nanmean(shade))
        # print(np.nanmean(wshade))
        skyshade = skyshade + shade
        # print(np.nanmean(skyshade))
        shade = None

    return skyshade


def shadows(in_array, az, alt, res):
    # Rows = i = y values, cols = j = x values
    rows = in_array.shape[0]
    cols = in_array.shape[1]
    shadow_array = np.zeros(in_array.shape)
    max_elev = np.max(in_array)
    max_distance = 1000

    az = 90. - az  # convert from 0 = north, cw to 0 = east, ccw

    azrad = az * np.pi / 180.
    altrad = alt * np.pi / 180.

    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            keep_going = True
            point_elev = in_array[i, j]  # the point we want to determine if in shadow
            # start calculating next point from the source point
            prev_i = i
            prev_j = j
            while keep_going:  # this inner loop loops through the possible values for each path
                # Figure out next point along the path
                delta_j = math.cos(azrad) * res
                delta_i = math.sin(azrad) * res
                next_i = prev_i + delta_i
                next_j = prev_j + delta_j
                # Update prev_i/j for next go-around
                prev_i = next_i
                prev_j = next_j

                # We need integar indexes for the array
                idx_i = int(round(next_i))
                idx_j = int(round(next_j))

                shadow = 1  # 0 if shadowed, 1 if not

                # distance for elevation check is distance from cell centers (idx_i/j), not distance along the path
                # critical height is the elevation that is directly in the path of the sun at given alt/az
                idx_distance = math.sqrt((i - idx_i)**2 + (j - idx_j)**2)
                critical_height = idx_distance * math.tan(altrad) * res + point_elev


                in_bounds = idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols
                in_height = critical_height < max_elev
                in_distance = idx_distance * res < max_distance

                if in_bounds and in_height and in_distance:
                # bounds check (array bounds, elevation check)
                # if idx_i >= 0 and idx_i < rows and idx_j >= 0 and idx_j < cols and critical_height < max_elev:
                    next_elev = in_array[idx_i, idx_j]
                    if next_elev > point_elev:  # only check if the next elev is greater than the point elev
                        if next_elev > critical_height:
                            shadow = 0
                            keep_going = False  # don't bother continuing to check

                else:
                    keep_going = False  # our next index would be out of bounds, we've reached the edge of the array

                print("{}: {}, {}".format(idx_distance, critical_height, max_distance))

            shadow_array[i, j] = shadow  # assign shadow value to output array
            # print("{}, {}: {}".format(i, j, shadow))


    return shadow_array


# variables
csv_path = r'C:\GIS\Data\Elevation\Uintahs\test2_nohdr.csv'
in_dem_path = r'C:\GIS\Data\Elevation\Uintahs\utest.tif'
out_dem_path = r'C:\GIS\Data\Elevation\Uintahs\utest_shadows2.tif'

gdal.UseExceptions()

lines = []
with open(csv_path, 'r') as l:
    reader = csv.reader(l)
    for line in reader:
        lines.append(line)
# options["lum_lines"] = lines

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

print("Reading array")
s_data = s_band.ReadAsArray()

# Close source file handle
s_band = None
s_fh = None

print("Processing array")
# sky = skymodel(s_data, lines)
# Test is 225 az, 25 alt
shad = shadows(s_data, 225, 25, cell_size)

out_array = shad

print("Writing output array")
# Set up target file in preparation for future writes
# If we've been given a vrt as a source, force the output to be geotiff
if driver.LongName == 'Virtual Raster':
    driver = gdal.GetDriverByName('gtiff')
if os.path.exists(out_dem_path):
    raise IOError("Output file {} already exists.".format(out_dem_path))

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
