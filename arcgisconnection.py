import logging
from functools import cached_property
from typing import List
from xml.etree.ElementTree import Element

from arcgis.features import FeatureLayer, Feature
from arcgis.geometry import Geometry
from shapely.geometry import Polygon

from kmlutilities import ReportProcessor


class ArcGisFeature:
    def __init__(self, feature: Feature):
        self.feature = feature

    @property
    def pidn(self) -> str:
        return self.feature.fields['PIDN']

    @property
    def address(self) -> str:
        return f"{self.feature.fields['PROPERTY_LOCATION_NUMBER']} {self.feature.fields['PROPERTY_LOCATION_STREET']} {self.feature.fields['PROPERTY_LOCATION_SUFFIX']} {self.feature.fields['PROPERTY_LOCATION_ZIP']}"

    @cached_property
    def geometry(self) -> Polygon:
        return Geometry(self.feature.geometry).as_shapely

    def intersects(self, other: Polygon) -> bool:
        return self.geometry.intersects(other)


class ArcGisProcessor:
    def __init__(self, layer_urls: List[str], layer_fields: List[str], doc: Element):
        self.__layer_urls = layer_urls
        self.__layer_fields = layer_fields
        self.kml_processor = ReportProcessor(doc)
        self.__doc = doc

    @cached_property
    def layers(self) -> list[FeatureLayer]:
        return [FeatureLayer(url=my_url) for my_url in self.__layer_urls]

    def load(self):
        # Connect to arcgis
        for layer in self.layers:
            logging.info(f"layer url={layer.url} properties={list(layer.properties)}")
            logging.info(f"layer url={layer.url} fields={layer.properties['fields']}")
            # Download the entire region we care about.
            layer_data = layer.query(out_fields=self.__layer_fields)
            features = [ArcGisFeature(feature) for feature in layer_data.features]
            # TODO - Parallelize this?
            for parcel in features:
                for trail in self.kml_processor.trail_geometry:
                    # Do the polygon intersection
                    if trail.intersects(parcel.geometry):
                        print(f"parcel: PIDN={parcel.pidn}, address={parcel.address}")

            # TODO - Write the list of parcels that we might be crossing.
            # TODO - Write metadata from those parcels into a report.
