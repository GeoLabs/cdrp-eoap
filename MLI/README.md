# OGC EOAP for ML Inference as a service

Description TBA

## Usage

First, build the required docker image and tag it with `geolabs/cdrp-mli-package:1.0`.

````
docker build . -t geolabs/cdrp-mli-package:1.0
````

Run the EOAP from the command line.

python -m app --dem https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/DEM/DEM_030.tif \
  --model https://www.geolabs.fr/dl/.RawData/M_ALL_006.hdf5 \
  --hillshade https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/DEM/DEM_030_HILLSHADE.tif  \
  --slope https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/DEM/DEM_030_SLOPE.tif  \
  --pre-image https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/fromOptical/JIUZ_PRE_S2_RGB_010_UINT8.tif \
  --post-image https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/fromOptical/JIUZ_POST_S2_RGB_010_UINT8.tif \
  --mask https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/noDataMask/SNOW_CLOUD_MASK_010.tif \
  --boundary https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/testBoundary/Test_006.tif

````
cwltool app.yml#geolabs_cdrp_hsa --item <YOUR_DEM_URL>
````

To deploy the EOAP on the ZOO-Project-DRU, use the following requests.

````
curl -X POST \
     -H "accept: application/json" \
     -H "Content-type: application/cwl+yaml" \
     -d @app.yml \
     <YOUR_ZOO-Project-DRU_URL>/processes
````

To execute the EOAP you can use the following request.

````
curl -X 'POST' \
  '<YOUR_ZOO-Project-DRU_URL>/processes/geolabs_cdrp_slope/execution' \
  -H 'accept: /*' \
  -H 'Prefer: respond-async;return=representation' \
  -H 'Content-Type: application/json' \
  -d '{
  "inputs": {
    "dem": "https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/DEM/DEM_030.tif",
    "model": "https://www.geolabs.fr/dl/.RawData/M_ALL_006.hdf5",
    "hillshade": "https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/DEM/DEM_030_HILLSHADE.tif",
    "slope": "https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/DEM/DEM_030_SLOPE.tif",
    "pre-image": "https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/fromOptical/JIUZ_PRE_S2_RGB_010_UINT8.tif",
    "post-image": "https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/fromOptical/JIUZ_POST_S2_RGB_010_UINT8.tif",
    "mask": "https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/noDataMask/SNOW_CLOUD_MASK_010.tif",
    "boundary": "https://www.geolabs.fr/dl/.RawData/AOI_E_S2-S2/testBoundary/Test_006.tif"
  }
}'
````

From the response, you can find the job identifier, which you can use to follow the process execution progress. Once the execution ends, you can access the result endpoint to access the produced stac catalog, which is a collection of hillshade, slope, and aspect features.


## References

[1] Barnes, Richard. 2016. RichDEM: Terrain Analysis Software. http://github.com/r-barnes/richdem