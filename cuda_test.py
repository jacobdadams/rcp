import csv
import math
import os
import sys
import time

import numba
import numpy as np
from numba import cuda
from osgeo import gdal


def sizeof_fmt(num, suffix="B"):
    """
    Quick-and-dirty method for formating file size, from Sridhar Ratnakumar,
    https://stackoverflow.com/questions/1094841/reusable-library-to-get-human-readable-version-of-file-size.
    """
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, "Yi", suffix)


def write_with_gdal(driver, out_path, out_array, nodata_value, transform, projection):
    # If we've been given a vrt as a source, force the output to be geotiff
    if driver.LongName == "Virtual Raster":
        driver = gdal.GetDriverByName("gtiff")

    lzw_opts = ["compress=lzw", "tiled=yes", "bigtiff=yes"]

    target_filehandle = driver.Create(out_path, cols, rows, bands, gdal.GDT_Float32, options=lzw_opts)
    target_filehandle.SetGeoTransform(transform)
    target_filehandle.SetProjection(projection)
    target_band = target_filehandle.GetRasterBand(1)
    target_band.SetNoDataValue(nodata_value)

    target_band.WriteArray(out_array)


def skymodel_on_device(input_array, luminance_lines, nodata, resolution):
    """
    Creates a unique hillshade based on a skymodel, implmenting the method
    defined in Kennelly and Stewart (2014), A Uniform Sky Illumination Model to
    Enhance Shading of Terrain and Urban Areas.

    in_array:       The input array, should be read using the supper_array
                    technique from below.
    lum_lines:      The azimuth, altitude, and weight for each iteration of the
                    hillshade. Stored as an array lines, with each line being
                    an array of [az, alt, weight].
    """

    # initialize skyshade as 0's
    skyshade = np.zeros(input_array.shape)

    # If it's all NoData, just return an array of 0's
    if input_array.mean() == nodata:
        return skyshade

    # Multiply elevation by 5 as per original paper
    input_array *= 5

    #: Configure device parameters
    threads = (16, 16)
    blockspergrid_x = math.ceil(input_array.shape[0] / threads[0])
    blockspergrid_y = math.ceil(input_array.shape[1] / threads[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    device_input_array = cuda.to_device(input_array)
    device_skyshade_output = cuda.device_array_like(input_array)

    # Loop through luminance file lines to calculate multiple hillshades
    for line in luminance_lines:
        azimuth = float(line[0])
        altitude = float(line[1])
        weight = float(line[2])

        #: Compute constants for hillshade
        azimuth = (90.0 - azimuth) * np.pi / 180.0
        altitude = altitude * np.pi / 180.0
        sin_azimuth = np.sin(azimuth)
        cos_azimuth = np.cos(azimuth)
        sin_altitude = np.sin(altitude)
        cos_altitude = np.cos(altitude)

        device_hillshade_output = cuda.device_array_like(input_array)

        hillshade_kernel[blockspergrid, threads](
            device_input_array, device_hillshade_output, cell_size, cos_azimuth, cos_altitude, sin_azimuth, sin_altitude
        )

        #: Compute constants for shadows
        delta_j = cos_azimuth
        delta_i = -1.0 * sin_azimuth
        tan_altitude = math.tan(altitude)
        max_steps = 600
        max_elevation = np.max(input_array)

        # precompute idx distances
        distances = np.ndarray((max_steps, 3), dtype=np.float32)
        for d in range(1, max_steps):
            distance = d * resolution
            step_height = distance * tan_altitude
            i_distance = delta_i * d
            j_distance = delta_j * d
            distances[d - 1] = (step_height, i_distance, j_distance)

        #: set up shadow arrays- start at 1 (not shadowed)
        shadow_output = np.ones(input_array.shape)
        device_shadow_output = cuda.to_device(shadow_output)
        device_distances = cuda.to_device(distances)

        shadows_kernel[blockspergrid, threads](
            device_input_array, device_shadow_output, device_distances, max_steps, max_elevation, nodata * 5
        )

        multiply_skyshade_kernel[blockspergrid, threads](
            device_hillshade_output, device_shadow_output, weight, device_skyshade_output
        )

    return device_skyshade_output.copy_to_host()


@cuda.jit
def hillshade_kernel(input_array, output_array, cell_size, cos_azimuth, cos_altitude, sin_azimuth, sin_altitude):
    i, j = cuda.grid(2)
    if i < input_array.shape[0] and j < input_array.shape[1]:
        #: gradient in both x and y dimensions
        #: get adjacent indices
        prev_i = i - 1
        next_i = i + 1
        prev_j = j - 1
        next_j = j + 1

        #: spacing is modified by cell size because input raster cell size is almost never 1 unit (meter/feet/whatever)
        spacing_i = 2 * cell_size
        spacing_j = 2 * cell_size

        #: modify for edge cases
        if prev_i < 0:
            prev_i = 0
            spacing_i = cell_size
        if next_i >= input_array.shape[0]:
            next_i = i
            spacing_i = cell_size
        if prev_j < 0:
            prev_j = 0
            spacing_j = cell_size
        if next_j >= input_array.shape[1]:
            next_j = j
            spacing_j = cell_size

        #: gradient is the difference between adjacent cells in x and y (modified for edges) divided by the spacing
        delta_x = (input_array[next_i, j] - input_array[prev_i, j]) / spacing_i
        delta_y = (input_array[i, next_j] - input_array[i, prev_j]) / spacing_j

        #: Calc rest of hillshade algorithm
        xx_plus_yy = delta_x * delta_x + delta_y * delta_y
        alpha = delta_y * cos_azimuth * cos_altitude - delta_x * sin_azimuth * cos_altitude
        shaded = (sin_altitude - alpha) / math.sqrt(1 + xx_plus_yy)

        output_array[i, j] = 127.5 * (shaded + 1)


@cuda.jit
def shadows_kernel(input_array, shadow_array, distances, max_steps, max_elevation, nodata):
    i, j = cuda.grid(2)
    if i < input_array.shape[0] and j < input_array.shape[1]:
        point_elev = input_array[i, j]  # the point we want to determine if in shadow

        for step in range(1, max_steps):  # start at a step of 1- a point cannot be shadowed by itself
            # No need to continue if it's already shadowed
            if shadow_array[i, j] == 0:
                # print("shadow break")
                break

            #: critical_height is the highest elevation at that point in the ray that would block the sun
            critical_height = distances[step - 1][0] + point_elev

            # idx_i/j are indices of array corresponding to current position + y/x distances
            idx_i = int(round(i + distances[step - 1][1]))
            idx_j = int(round(j + distances[step - 1][2]))

            in_bounds = idx_i >= 0 and idx_i < input_array.shape[0] and idx_j >= 0 and idx_j < input_array.shape[1]
            in_height = critical_height < max_elevation

            if in_bounds and in_height:
                next_elev = input_array[idx_i, idx_j]
                # Bail out if we hit a nodata area
                if next_elev == nodata or next_elev == np.nan:
                    break

                if next_elev > point_elev and next_elev > critical_height:
                    shadow_array[i, j] = 0

                    #: TODO: may need to think about synching here for the earlier check if it's already shadowed
                    # set all array indices in between our found shadowing index and the source index to shadowed
                    #: don't need to check bounds because the previous in_bounds check ensures that all indices
                    #: calculated up to this step are in bounds (???)
                    for step2 in range(1, step):
                        i2 = int(round(i + distances[step2 - 1][1]))
                        j2 = int(round(j + distances[step2 - 1][2]))
                        shadow_array[i2, j2] = 0

                    break  # We're done with this point, move on to the next


@cuda.jit
def multiply_skyshade_kernel(hillshade_array, shadow_array, weight, output_array):
    i, j = cuda.grid(2)
    if i < hillshade_array.shape[0] and j < hillshade_array.shape[1]:
        output_array[i, j] = output_array[i, j] + (hillshade_array[i, j] * shadow_array[i, j] * weight)


if __name__ == "__main__":
    start = time.time()
    in_dem_path = r"C:\gis\Elevation\PG\bh12TVK3600068000.img"
    # normal_out_path = r"C:\gis\Elevation\PG\pg_hs-normal.tif"
    # cuda_out_path = r"C:\gis\Elevation\PG\pg_hs-cuda.tif"

    # if os.path.exists(normal_out_path):
    #     raise IOError("Output file {} already exists.".format(normal_out_path))
    # if os.path.exists(cuda_out_path):
    #     raise IOError("Output file {} already exists.".format(cuda_out_path))

    source_filehandle = gdal.Open(in_dem_path, gdal.GA_ReadOnly)
    rows = source_filehandle.RasterYSize
    cols = source_filehandle.RasterXSize
    driver = source_filehandle.GetDriver()
    bands = source_filehandle.RasterCount
    source_band = source_filehandle.GetRasterBand(1)

    # Get source georeference info
    transform = source_filehandle.GetGeoTransform()
    projection = source_filehandle.GetProjection()
    cell_size = abs(transform[5])  # Assumes square pixels where height=width
    source_nodata_value = source_band.GetNoDataValue()

    print("Reading array...")
    source_data = source_band.ReadAsArray()

    print("source_data: {}".format(sizeof_fmt(sys.getsizeof(source_data))))

    # luminance_lines = []
    # with open(r"C:\gis\git\rcp\turbid_315_15_250_nohdr.csv", "r") as lines_file:
    #     reader = csv.reader(lines_file)
    #     for line in reader:
    #         luminance_lines.append([float(line[0]), float(line[1]), float(line[2])])
    luminance_lines = [[315, 15, 0.5], [45, 15, 0.5]]

    print("running skymodel...")
    skymodel = skymodel_on_device(source_data, luminance_lines, source_nodata_value, cell_size)

    write_with_gdal(
        driver, r"C:\gis\Elevation\PG\pg_skymodel-cuda-1.tif", skymodel, source_nodata_value, transform, projection
    )
    print(f"Time: {time.time() - start}")
