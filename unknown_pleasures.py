import math
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np


# =========== TODO ===========
# * Properly scale x-axis according to map units 
# * Specify num_cols/rows instead of x/y_gap
# * Compute bounding box of read area and change raster read method to only read this area
#       # Combine x and y into one loop (for sake of clean code)
#       As generating x and y coords, keep track of min and max x and y (becomes bounding box)
#       Do the x/y coord sys to row/col conversion for the min x/y for read origin
#       change source_x/y_origin to min x/y values (used in x/y -> row/col translation for rows of points
#       ReadAsArray(x_off, y_off, x_max-x_min, y_max-y_min)
# * Create polygon of view area, lay over dem or hillshade?
# * Change our offset to matplotlib's offset
#   * Then, change facecolor to a gradient to add snowcaps or other elevation-dependent colors

# in_dem_path = "c:\\gis\\data\\elevation\\ZionsNED10m\\zions_dem.tif"
in_dem_path = "c:\\gis\\data\\elevation\\SaltLake10mDEM\\sl10m.tif"
# in_dem_path = "c:\\gis\\data\\elevation\\tetonyellowstone10m\\tydem10m.tif"

gdal.UseExceptions()

# Get source file metadata (dimensions, driver, proj, cell size, nodata)
s_fh = gdal.Open(in_dem_path, gdal.GA_ReadOnly)
rows = s_fh.RasterYSize
cols = s_fh.RasterXSize
driver = s_fh.GetDriver()
s_band = s_fh.GetRasterBand(1)
nodata = s_band.GetNoDataValue()
# Get source georeference info
transform = s_fh.GetGeoTransform()
projection = s_fh.GetProjection()
#cell_size = abs(transform[5])  # Assumes square pixels where height=width
#s_nodata = s_band.GetNoDataValue()

source_x_origin = transform[0]
source_y_origin = transform[3]
pixel_width = transform[1]
pixel_height = -transform[5]
#print("x orig: {} y orig: {} width: {} height: {}".format(source_x_origin, source_y_origin, pixel_width, pixel_height))

data = s_band.ReadAsArray(0, 0, cols, rows)
masked_data = np.ma.masked_equal(data, nodata)
del data

data_min = masked_data.min()

# x vals = cols = j, y vals = rows = i
# coords are in x, y (easting, northing)
# loop through rows- every 100 feet
#   loop through each column- every 100 feet
# orig drawing has 80 lines
# UL: 1,467,208, 3,837,848
# LR: 1,594,422, 3,756,858
# x between 1,594,000 and 1,467,000 => 127 @ 1000'
# y between 3,837,000 and 3,756,000 => 81 @ 1000'

# Original
# read_x_right =   1594000
# read_x_left =    1467000
# read_y_top =     3837000
# read_y_bottom =  3756000

# Extend all
# read_x_right =   1625000
# read_x_left =    1467000
# read_y_top =     3870000
# read_y_bottom =  3710000

# # Extend X
# read_x_right =   1625000
# read_x_left =    1467000
# read_y_top =     3837000
# read_y_bottom =  3756000
#
# # horizontal gap
# x_gap = 500
# # vertical gap
# y_gap = 1000

## original unrotated method
# # A list of lists of elevations x_gap apart horizontally. Each sub-list is y_gap apart vertically
# row_elevs_list = []
#
# # coordinates: easting = x = cols, northing = y = rows
#
# # Loop through rows (y), then each column (x) in each row (in supplied coords)
# # add x/y_gap to range to add the last value to the range
# for y in range(read_y_bottom, read_y_top + y_gap, y_gap):
#     row_elevs = []  # each entry should be elevation at that one point
#     for x in range(read_x_left, read_x_right + x_gap, x_gap):
#
#         # Get the raster indexes of the supplied coords
#         # x is x coordinate, y is y coordinate in supplied coord system
#         source_x_index = int((x - source_x_origin) / pixel_width)
#         source_y_index = int((source_y_origin - y) / pixel_height)
#
#         # Read from raster, which is accessed via [row, col]
#         elev = data[source_y_index, source_x_index]
#         row_elevs.append(elev)
#     row_elevs_list.append(row_elevs)

# set by number of rows/cols desired instead of specifying absolute extent
# num_rows = int(read_y_top - read_y_bottom) / y_gap
# num_cols = int(read_x_right - read_x_left) / x_gap
#
# rotate = 5.0
# DEG_TO_RAD = math.pi / 180.0
# rotate_rad = rotate * DEG_TO_RAD

# TEMP FOR TESTING, CAN DELETE
# num_rows = 10
# num_cols = 10



# Lower y_gap -> higher 'viewpoint'
# print_offset acts as a scaling factor. The higher the offset, the less pronounced the terrain
#   200 seems a little excessive, 2000 is very flat

# Cache Front
# Origin x: 1,535,295
# Origin y: 3,870,610
# Rotate: 90

# cache_front
# origin = (1535295, 3870610)
# rotate = 90
# width = 140000
# height = 50000

# # Wellsvilles
# origin = (1538200, 3738970)
# rotate = 270
# width = 80000
# height = 80000

# valley south
# origin = (1619695, 3840010)
# rotate = 180
# width = 160000
# height = 140000

#paranauweep
#origin = (324539, 4115968)
# Pweep 2
# origin = (322237, 4116722)
# rotate = 80
# width = 6500
# height = 17000

# # Main Zion's canyon
# origin = (326574, 4125965)
# rotate = 0
# width = 1500
# height = 4500

# Wasatch Front
# origin = (419938, 4515057)
# rotate = 90
# width = 35000
# height = 50000
#Wasatch Front Extended
# origin = (406550, 4518541)
# rotate = 70
# width = 60000
# height = 40000
# origin = (413669, 4504336)
# rotate = 70
# width = 55000
# height = 40000
#Wasatch Front Centered
origin = (413669, 4504336)
rotate = 75
width = 45000
height = 60000

# Tetons
# origin = (517390, 4805442)
# rotate = 290
# width = 50000
# height = 20000
# Grand Teton
# origin = (522823, 4829950)
# rotate = 290
# width = 25000
# height = 15000

# LCC down
# origin = (447944, 4488521)
# rotate = 270
# width = 7500
# height = 20000

# Wasatch Back
# origin = (476730, 4456888)
# rotate = 265
# width = 70000
# height = 50000

# Provo Cabin
#origin = (490117, 4487813)
#rotate = 225
#width = 4500
#height = 4000

# smaller offsett = larger vertical scaling, "lower" viewpoint
print_offset = 80
grey_max = 1

# horizontal gap
x_gap = 500
# vertical gap
y_gap = 300

num_rows = int(height / y_gap)
num_cols = int(width / x_gap)

# Perspective seems to be a function of the print offset. low offset = looking from the ground. high offset = airplane view

print("Rows: {}    Cols: {}".format(num_rows, num_cols))

# ===== Generating Sample Point X and Y Coordinates =====

# starting y value for nth row: previous starting y value + cos(theta)*y_gap
#   next y value in nth row: previous y value - cos(90-theta)*x_gap
# starting x value for nth row: previous starting x value + sin(theta)*y_gap
#   next x value in nth row: previous x value + sin(90-theta)*x_gap

# list of list of (y,x) tuples, each sublist is a row
# y, x in coord system
coord_rows = []  

row_y_origin = origin[1]
row_x_origin = origin[0]

# Track the min/max x/y; initial points are the origin
y_min = origin[1]
y_max = origin[1]
x_min = origin[0]
x_max = origin[0]

for row in range(0, num_rows):  # build list of y,x-values in coord system for each row

    # list of (y,x) vals in coord system for this row
    current_row = []
    # first y, x is the row origin points
    current_row.append((row_y_origin, row_x_origin))  
    
    # Set for next row
    prev_y_val = row_y_origin
    prev_x_val = row_x_origin
    
    # calculate the next y, x values for each column in this row
    for col in range(1, num_cols):  # start at 1 because we already added the origin
        y_val = prev_y_val - math.cos(math.radians(90 - rotate)) * x_gap
          
        x_val = prev_x_val + math.sin(math.radians(90 - rotate)) * x_gap
        
        current_row.append((y_val, x_val))
        
        prev_x_val = x_val  # set the x val for the next col in this row
        prev_y_val = y_val  # set the y val for the next col in this row
    
        # min/max values
        if y_val < y_min:
            y_min = y_val
        elif y_val > y_max:
            y_max = y_val
            
        if x_val < x_min:
            x_min = x_val
        elif x_val > x_max:
            x_max = x_val
    
    # add this row's (y,x) list to the list of rows
    coord_rows.append(current_row)
    
    # Set the y value for the next row
    row_y_origin = row_y_origin + math.cos(math.radians(rotate)) * y_gap
    row_x_origin = row_x_origin + math.sin(math.radians(rotate)) * y_gap    

    
# ===== Get the Actual Elevations from the Raster Array =====
# A list of lists of elevations x_gap apart horizontally. Each sub-list is y_gap apart vertically. Rotation was accomplished when the lists of x and y points in coord sys were generated
row_elevs_list = []

# coordinates: easting = x = cols, northing = y = rows

# Loop through list of coordinate rows (each item is a (y,x) tuple)
for row in coord_rows:
    row_elevs = []
    for coord_pair in row:
        y = coord_pair[0]
        x = coord_pair[1]
        # Get the raster indexes of the supplied coords
        # x is x coordinate, y is y coordinate in supplied coord system
        source_x_index = int((x - source_x_origin) / pixel_width)
        source_y_index = int((source_y_origin - y) / pixel_height)

        # Read from raster, which is accessed via [row, col]
        try:
            elev = masked_data[source_y_index, source_x_index]
            if elev < data_min:
                row_elevs.append(data_min)
            else:
                row_elevs.append(elev)
        except IndexError:
            row_elevs.append(data_min)
    row_elevs_list.append(row_elevs)


#elevs = [r for r in row_elevs_list[0]]

# shift each row of elevations up by i * print_offset for printing
#   which makes it print_offset higher than the row before it
# Maybe look at using matplotlib offsets rather than modifying the data. Then we might be able to add a gradient to the elevation facecolor, doing whitecapped mountains or something like that above a certain point.
new_row_elevs_list = []
for i, row in enumerate(row_elevs_list):
    offset = i * print_offset
    new_row_elevs = []
    for val in row:
        new_row_elevs.append(val + offset)
    new_row_elevs_list.append(new_row_elevs)

del masked_data        
    
# Print it out
y_start = row_elevs_list[0][0]
x_vals = range(0, len(new_row_elevs_list[0])) # first collection of x-values used as x-axis, should probably get values from data- ???
for i, stripe in enumerate(new_row_elevs_list[::-1]):
    # Trying to set the color as a gradient fading to grey in the back
    # color = '0.75' = 75% grey
    # Take our scaling factor and raise it to a power, so that it stays darker for longer and then goes quickly to white- somewhere between 2 to 4 or 5
    grey = math.pow(grey_max/num_rows * (len(new_row_elevs_list) - i), 4)
    color = str(grey)
    # Working from the back, add the polygon and the line for each slice
    plt.fill_between(x_vals, stripe, facecolor = 'white', edgecolor = color)
    #plt.plot(stripe, color='white')
# Try to set the ymin to be print_offset below the lowest elevation
min_elev = min(new_row_elevs_list[0])  # this works becuase the front slice hides everything behind it... but we can get cooler than that

# from https://dbader.org/blog/python-min-max-and-nested-lists, https://stackoverflow.com/questions/33269530/get-max-value-from-a-list-with-lists
# min_elev = min(map(lambda x: min(x), row_elevs_list)) #  this gets the minimum elevation value found, which could lead to undesired results if the foreground is not the lowest elevation (looking down canyon instead of up, for example). So we'll just go with the first method
plt.ylim(ymin = min_elev - print_offset)
plt.axis('off')
plt.show()

print("y min: {}\ty max: {}".format(y_min, y_max))
print("x min: {}\tx max: {}".format(x_min, x_max))

#print(row_list[0])
#print(row_elevs_list[0])

# for reading given a list of points in the same coords as the source_file:
# for point in points_list:
#     col = int((point[0] - source_x_origin) / pixel_width)
#     row = int((source_y_origin - point[1]) / pixel_height)
#
#     print("{}, {}, {}".format(row, col, data[row][col]))

# Close source file handle
s_band = None
s_fh = None
