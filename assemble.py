import os
import numpy as np
from osgeo import gdal

# Path to allan submission
PATH_A = "C:/Users/vincent/Documents/flair/assemblage/alan/results/"
# Path to vincent submission
PATH_V = "C:/Users/vincent/Documents/flair/assemblage/vincent/"

# final submission folder
OUTPUT = "C:/Users/vincent/Documents/flair/predictions/"

if not os.path.exists(OUTPUT):
    os.makedirs(OUTPUT)

FILES = os.listdir(PATH_A)

# best score for vincent submission
classe_vincent = [2,3,4,6,8,11]
# best score for alan submission
classe_alan = [0,1,5,7,9,10]

for img in FILES:
    arr = np.ones(shape=(512,512)) * 12
    ds = gdal.Open(PATH_V + img)
    v = ds.ReadAsArray()
    a = gdal.Open(PATH_A + img).ReadAsArray()
    arr = np.where(np.isin(v, classe_vincent), v, arr)
    arr = np.where(np.isin(a, classe_alan), a, arr)

    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(OUTPUT + img, 512, 512, 1, gdal.GDT_Byte)
    dataset.GetRasterBand(1).WriteArray(arr)
    proj = ds.GetProjection() #you can get from a exsited tif or import 
    dataset.SetGeoTransform(list(ds.GetGeoTransform() ))
    dataset.SetProjection(proj)
    dataset.FlushCache()
    dataset=None
    a = v = ds = None