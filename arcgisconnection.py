import math
import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)

import logging
from functools import cached_property, cache
from typing import List, Tuple, Dict

import pandas as pd
import multiprocessing as mp
from arcgis.features import FeatureLayer, Feature
from arcgis.geometry import Geometry
from shapely.geometry import Polygon, LineString

from kmlutilities import ReportProcessor, geodetic_to_ECEF, ECEF_to_ENU, parse_coordinates_tuple


class ArcGisFeature:
    def __init__(self, feature: Feature, property_id_key: str, address_name_map: List[str]):
        self.feature = feature
        self.property_id_key = property_id_key
        self.address_name_map = address_name_map

    @property
    def property_id(self) -> str:
        return self.feature.attributes[self.property_id_key]

    @property
    def address(self) -> str:
        return " ".join([self.feature.attributes[address_key] for address_key in self.address_name_map])

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
    def __init__(self, layer_urls: Dict[str, str], property_id_map: Dict[str, str], address_name_map: Dict[str, List[str]], doc):
        self.__layer_urls = layer_urls
        self.__property_id_map = property_id_map
        self.__address_name_map = address_name_map
        self.kml_processor = ReportProcessor(doc)
        self.__doc = doc

    @cached_property
    def layers(self) -> list[FeatureLayer]:
        return [FeatureLayer(url=my_url) for my_url in self.__layer_urls]

    @cache
    def layer_fields(self, layer_name: str) -> List[str]:
        return [self.__property_id_map[layer_name], *self.__address_name_map[layer_name]]

    def write_touched_properties_report(self, report_file: str) -> None:
        # Connect to arcgis
        for layer_name, layer_url in self.__layer_urls.items():
            layer = FeatureLayer(url=layer_url)
            logging.info(f"layer url={layer.url} properties={list(layer.properties)}")
            logging.info(f"layer url={layer.url} fields={layer.properties['fields']}")
            # Download the entire region we care about.
            # TODO - Cache this?
            layer_data = layer.query(out_fields=self.layer_fields(layer_name))  # , return_all_records=False, result_record_count=1000)
            features = [ArcGisFeature(feature, self.__property_id_map[layer_name], self.__address_name_map[layer_name]) for feature in layer_data.features]

            # Parallelize this?
            check_properties: List[Tuple[str, LineString, ArcGisFeature]] = []
            for parcel in features:
                for trail_name, trail_line in self.kml_processor.trail_geometry.items():
                    check_properties.append((trail_name, trail_line, parcel))

            pool = mp.Pool(mp.cpu_count() // 2)  # TODO - Skip logical processors correctly.
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
