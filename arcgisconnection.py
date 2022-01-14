import math
import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
from functools import cached_property
from typing import List, Tuple
from xml.etree.ElementTree import Element

import pandas as pd
import multiprocessing as mp
from arcgis.features import FeatureLayer, Feature
from arcgis.geometry import Geometry
from shapely.geometry import Polygon, LineString

from kmlutilities import ReportProcessor, geodetic_to_ECEF, ECEF_to_ENU, parse_coordinates_tuple


class ArcGisFeature:
    def __init__(self, feature: Feature):
        self.feature = feature

    @property
    def pidn(self) -> str:
        return self.feature.attributes['PIDN']

    @property
    def address(self) -> str:
        return f"{self.feature.attributes['PROPERTY_LOCATION_NUMBER']} {self.feature.attributes['PROPERTY_LOCATION_STREET']} {self.feature.attributes['PROPERTY_LOCATION_SUFFIX']} {self.feature.attributes['PROPERTY_LOCATION_ZIP']}"

    @cached_property
    def geometry(self) -> Polygon:
        # Convert from world reference to reser-reference
        geo = Geometry(self.feature.geometry)
        lon_lat = geo.project_as(spatial_reference="4326", transformation_name="transformation")
        assert len(lon_lat.rings) == 1
        return Polygon(ECEF_to_ENU(geodetic_to_ECEF(parse_coordinates_tuple(lon_lat.rings[0]))))

    def intersects(self, other: Polygon) -> bool:
        return self.geometry.intersects(other)


def check_for_intersection(trail_name, trail_line, parcel):
    # Do the polygon intersection
    return trail_line.intersects(parcel.geometry)


class ArcGisProcessor:
    def __init__(self, layer_urls: List[str], layer_fields: List[str], doc: Element):
        self.__layer_urls = layer_urls
        self.__layer_fields = layer_fields
        self.kml_processor = ReportProcessor(doc)
        self.__doc = doc

    @cached_property
    def layers(self) -> list[FeatureLayer]:
        return [FeatureLayer(url=my_url) for my_url in self.__layer_urls]

    def write_touched_properties_report(self, report_file: str) -> None:
        # Connect to arcgis
        for layer in self.layers:
            logging.info(f"layer url={layer.url} properties={list(layer.properties)}")
            logging.info(f"layer url={layer.url} fields={layer.properties['fields']}")
            # Download the entire region we care about.
            # TODO - Cache this?
            layer_data = layer.query(out_fields=self.__layer_fields) # , return_all_records=False, result_record_count=1000)
            features = [ArcGisFeature(feature) for feature in layer_data.features]

            # Parallelize this?
            check_properties: List[Tuple[str, LineString, ArcGisFeature]] = []
            for parcel in features:
                for trail_name, trail_line in self.kml_processor.trail_geometry.items():
                    check_properties.append((trail_name, trail_line, parcel))

            pool = mp.Pool(mp.cpu_count())
            touches_property = pool.starmap(check_for_intersection, check_properties, chunksize=math.ceil(len(check_properties)/mp.cpu_count()))
            pool.close()

            logging.info(f"Trail/Parcel matching completed. Match count={np.count_nonzero(np.array(touches_property))}")
            report_columns = ['Trail Name']
            field_names = check_properties[0][2].feature.fields
            report_columns.extend(field_names)  # field names are consistent
            report_list = []
            # Write metadata from those parcels into a report.
            for ij, touched_property in enumerate(check_properties):
                if not touches_property[ij]:
                    continue
                property_attributes = touched_property[2].feature.as_dict['attributes']
                row_values = [property_attributes[name] for name in field_names]
                report_list.append([touched_property[0], *row_values])
            report_data = pd.DataFrame(data=report_list, columns=report_columns)

            report_data.to_csv(report_file, index=False)
