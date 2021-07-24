import numpy as np

from rcp import methods

def test_basic_shadowing():
    r = np.arange(5)
    d = np.minimum(r,r[::-1])
    test_array = np.minimum.outer(d,d)

    shadows = methods.shadows(test_array, az=315, alt=45, res=1, overlap=0, nodata=-1)
    test_shadows = np.array(
        [
            [1,1,1,1,1],
            [1,1,1,1,1],
            [1,1,1,1,0],
            [1,1,1,1,0],
            [1,1,0,0,0],
        ]
    )
    assert np.array_equal(shadows, test_shadows)

def test_shadowing_respects_overlap():
    r = np.arange(start=0, stop=24, step=4)
    d = np.minimum(r,r[::-1])
    test_array = np.minimum.outer(d,d)

    shadows = methods.shadows(test_array, az=315, alt=45, res=1, overlap=1, nodata=-1)
    test_shadows = np.array(
        [
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,1,1],
            [1,1,1,1,0,1],
            [1,1,1,0,0,1],
            [1,1,1,1,1,1],
        ]
    )
    assert np.array_equal(shadows, test_shadows)