#*****************************************************************************
# 
#  Project:  Contour Trimming
#  Purpose:  Identify short-length contours that are peaks, not spurious and
#            annoying contours left over from high-res source DEM
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

import arcpy

fc = r'I:\jadams.gdb\Elevation\contours_md506050_5ft_smoothed_80'
fields = ['OID@', 'elev', 'SHAPE@LENGTH']

contour_layer = "contour_layer"

arcpy.da.MakeFeatureLayer_managment(fc, contour_layer)

with arcpy.da.UpdateCursor(fc, fields) as everything:
    for row in everything:
        if row[2] < 500:  # If it's a small contour, check if it's higher than others
            where = "OBJECTID = {}".format(row[0])
            arcpy.da.MakeFeatureLayer_management(fc, "temp_contour_layer", where)
            arcpy.SelectLayerByLocation_management(contour_layer,
                                                   "WITHIN_A_DISTANCE",
                                                   "temp_contour_layer",
                                                   100, "NEW_SELECTION")
            counter = 0
            elev = 0
            # I hope that search cusors respect selections....
            with arcpy.da.SearchCursor("temp_contour_layer", "elev") as nearby_cursor:
                for row in nearby_cursor:
                    counter += 1
                    elev += row[0]
            
