# google-maps-analysis
Google Maps Analysis
Analyze your custom google maps for various trails and regions. It outputs summary reports with reporting region polygons defined. To define a report polygon, have a table column `official` with value `REPORT` (case insensitive). Summary pivot table columns can also be defined as below. Extra reporting csv files will be created as needed for the pivot tables and report region polygons.
Example script parameters:
```
traillength
--kmlfile
"C:/Users/scott/downloads/Jason Official.kml"
--reportfile
"C:/Users/scott/downloads/Jason Official_length.csv"
--summarycolumns
"official"
```

