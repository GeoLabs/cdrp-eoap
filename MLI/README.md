# OGC EOAP for ML Inference as a service

Description TBA

## Usage

First, build the required docker image and tag it with `geolabs/cdrp-mli-package:1.0`.

````
docker build . -t geolabs/cdrp-mli-package:1.0
````

Run the EOAP from the command line.

python -m app --dem <dem_data_file> \
  --model https://www.geolabs.fr/dl/.RawData/M_ALL_006.hdf5 \  #Model file
  --hillshade <hill_shade_data>  \
  --slope <slope_data>  \
  --pre-image <pre_event_image> \
  --post-image <post_event_image> \
  --mask <mask_for_cloud_snow> \
  --boundary <aoi_boundary>

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
    "dem": "<dem_data_file>",
    "model": "https://www.geolabs.fr/dl/.RawData/M_ALL_006.hdf5",
    "hillshade": "<hill_shade_file>",
    "slope": "<slope_data_file>",
    "pre-image": "<pre_event_file>",
    "post-image": "<post_image_file>",
    "mask": "<mask_data>",
    "boundary": "<aoi_boundary>"
  }
}'
````

From the response, you can find the job identifier, which you can use to follow the process execution progress. Once the execution ends, you can access the result endpoint to access the produced stac catalog, which is a collection of hillshade, slope, and aspect features.


## References

[1] Barnes, Richard. 2016. RichDEM: Terrain Analysis Software. http://github.com/r-barnes/richdem
