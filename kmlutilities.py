import datetime
import logging
from typing import List, Dict, Tuple, Generator

import gpxpy as gpxpy
import lxml.etree
import numpy as np
import pandas as pd
import pykml.parser
import shapely.geometry
from shapely.geometry import LineString, Polygon
from pykml.factory import KML_ElementMaker as KML


class ReportProcessor:
    trail_columns = ['Trail Name', 'Length [mile]']
    region_columns = ['Region Name', 'Area [mi^2]']

    def __init__(self, polygon_format="kml"):
        self.polygon_format = polygon_format
        self.trail_columns = ReportProcessor.trail_columns
        self.region_columns = ReportProcessor.region_columns
        self.trail_data: pd.DataFrame = None
        self.report_data: pd.DataFrame = None
        self.trail_lines: List['Placemark'] = None
        self.report_polygons: List['Placemark'] = None
        self.trails_report: Dict[str, pd.DataFrame] = dict()
        self.report_trail_geo: Dict[str, List] = dict()

        self.kml_styles: List = None
        self.kml_style_maps: List = None

    def generate_report(self, summary_columns: List[str], report_file: str) -> None:
        self.create_pivot_table_summaries(summary_columns, report_file)
        self.write_report_files(report_file)

    def load(self, doc) -> None:
        self.kml_styles = [style for style in doc.Document.Style]
        self.kml_style_maps = [stylemap for stylemap in doc.Document.StyleMap]
        extended_columns = get_extended_data(doc.Document.Placemark[0].ExtendedData).keys()
        self.trail_columns.extend(extended_columns)
        self.trail_data = pd.DataFrame(columns=self.trail_columns)
        self.report_data = pd.DataFrame(columns=self.region_columns)

        self.trail_lines = [placemark for placemark in doc.Document.Placemark if hasattr(placemark, 'LineString')]
        # self.trail_lines.extend([placemark for placemark in doc.Document.Placemark
        #                    if hasattr(placemark, 'Polygon') and not is_report(placemark)])
        self.report_polygons: List = [placemark for placemark in doc.Document.Placemark
                                      if hasattr(placemark, 'Polygon') and is_report(placemark)]

        self.trail_data, _ = self.__calculate_trail_data()
        self.__calculate_report_data()

    def create_pivot_table_summaries(self, summary_columns: List[str], report_file: str) -> None:
        # Summarize by type, status, and official
        for column in summary_columns:
            self.__summary_pivot_table(column, report_file)

    def write_report_files(self, report_file: str) -> None:
        # Report polygons
        self.report_data.to_csv(report_file.replace('.csv', f'_report_polygon.csv'), index=False)
        self.trail_data.to_csv(report_file, index=False)
        for (report_name, report) in self.trails_report.items():
            report.to_csv(report_file.replace('.csv', f'_{report_name}.csv'), index=False)
            # Write the KML file
            self.write_polygon_file(report_file, report_name)

    def write_polygon_file(self, report_file, report_name):
        if self.polygon_format == "kml":
            self.__write_kml_file(report_name, report_file.replace(".csv", f"_{report_name}.kml"))
        elif self.polygon_format == "gpx":
            self.__write_gpx_file(report_name, report_file.replace(".csv", f"_{report_name}.gpx"))
        else:
            raise NotImplementedError(f"Unsupported polygon format {self.polygon_format}")

    def __write_kml_file(self, report_name: str, output_file_name: str) -> None:
        report_kml = KML.kml(
            KML.Document(
                KML.name(report_name),
                *self.kml_styles,
                *self.kml_style_maps,
                *self.report_trail_geo[report_name]
            )
        )
        with open(output_file_name, 'wb') as fid:
            fid.write(lxml.etree.tostring(report_kml, pretty_print=True))

    def __write_gpx_file(self, report_name: str, output_file_name: str) -> None:
        report_gpx = gpxpy.gpx.GPX()
        report_gpx.tracks.extend(self.report_trail_geo[report_name])

        with open(output_file_name, 'w') as fid:
            fid.write(report_gpx.to_xml(prettyprint=True))

    def __calculate_report_data(self) -> None:
        for report in self.report_polygons:
            report_name = report.name.text.strip()
            logging.info(f'Polygon: {report_name}')
            report_poly = get_shapely_shape(report)
            row = [report_name, report_poly.area / 2589988.110336]
            self.report_data = self.report_data.append(pd.DataFrame([row], columns=self.report_data.columns))

            # Trim to shape and report
            report, lines = self.__calculate_trail_data(report_poly)
            self.trails_report[report_name] = report
            self.report_trail_geo[report_name] = lines

    def __calculate_trail_data(self, trim_poly=None) -> Tuple[pd.DataFrame, List]:
        data = pd.DataFrame(columns=ReportProcessor.trail_columns)
        lines = list()
        for ij, placemark in enumerate(self.trail_lines):
            placemark_name = placemark.name.text.strip()
            logging.info(f'Trail: {placemark_name} \\ {trim_poly}')
            ext_data = get_extended_data(placemark.ExtendedData)
            shapely_trail: LineString = get_shapely_shape(placemark)
            shapely_trail = shapely_trail.intersection(trim_poly) if trim_poly else shapely_trail
            distance = shapely_trail.length / 1609  # m -> mile
            if distance == 0.0:
                continue

            row = [placemark_name, distance]
            row.extend(ext_data.values())
            data = data.append(pd.DataFrame([row], columns=data.columns))

            # Generate the new placemark object
            if trim_poly and distance > 0.0:
                track = self.create_track(placemark, placemark_name, shapely_trail)
                lines.append(track)

        return data, lines

    def create_track(self, placemark, placemark_name, shapely_trail):
        if self.polygon_format == "kml":
            track = KML.Placemark(
                KML.name(placemark_name),
                KML.description(""),
                placemark.styleUrl,
                placemark.ExtendedData,
                KML.LineString(
                    KML.tessellate(1),
                    KML.coordinates(
                        get_kml_linestring(shapely_trail)
                    )
                )
            )
        elif self.polygon_format == "gpx":
            track = gpxpy.gpx.GPXTrack()
            track.name = placemark_name
            segment = gpxpy.gpx.GPXTrackSegment()
            track.segments.append(segment)
            points = get_geodetic_coordinates(shapely_trail)
            for jk in range(points.shape[0]):
                lon = points[jk, 0]
                lat = points[jk, 1]
                alt = points[jk, 2]
                segment.points.append(gpxpy.gpx.GPXTrackPoint(lat, lon, alt, time=next(self.__time_stamp())))
        else:
            raise NotImplementedError(f"Unsupported polygon format {self.polygon_format}")
        return track

    def __summary_pivot_table(self, pivot_column: str, report_file: str, data_column: str = 'Length [mile]') -> None:
        official = self.trail_data.pivot_table(index=pivot_column, values=data_column, aggfunc='sum')
        official = official.rename({'': '[BLANK]'})

        official.to_csv(report_file.replace('.csv', f'_pivot_{pivot_column}.csv'))

    @staticmethod
    def __time_stamp() -> Generator:
        start_time = datetime.datetime.now()
        while True:
            yield start_time
            start_time += datetime.timedelta(seconds=1)


def get_shapely_shape(placemark):
    if hasattr(placemark, 'LineString'):
        return LineString(ECEF_to_ENU(geodetic_to_ECEF(parse_coordinates(placemark.LineString.coordinates))))
    elif hasattr(placemark, 'Polygon'):
        return Polygon(
            ECEF_to_ENU(geodetic_to_ECEF(parse_coordinates(placemark.Polygon.outerBoundaryIs.LinearRing.coordinates))))
    else:
        raise NotImplementedError


def get_kml_linestring(line: LineString) -> str:
    geodetic_coordinates = get_geodetic_coordinates(line)
    point_strings = []
    for ij in range(geodetic_coordinates.shape[0]):
        coords = geodetic_coordinates[ij, :]
        point_strings.append(f"{coords[0]},{coords[1]},{coords[2]}")
    return "\n".join(point_strings)


def get_geodetic_coordinates(line):
    if type(line) is shapely.geometry.MultiLineString:
        coords = np.vstack([np.array(line.coords) for line in line.geoms])
    else:
        coords = np.array(line.coords)
    geodetic_coordinates = ECEF_to_geodetic(ENU_to_ECEF(coords))
    return geodetic_coordinates


def parse_coordinates(coordinates) -> np.ndarray:
    # Lat, Lon, h
    coord_strings = split_string(coordinates.text)
    coords = np.zeros((len(coord_strings), 3))
    for ij in range(len(coord_strings)):
        coords[ij, :] = geo_coords(coord_strings[ij])
    return coords


def ecef_constants():
    a = 6378137.0  # m
    b = 6356752.3  # m
    e2 = 1 - b ** 2 / a ** 2
    return a, b, e2


def geodetic_to_ECEF(coords: np.ndarray) -> np.ndarray:
    a, b, e2 = ecef_constants()
    N_phi = lambda phi: a / np.sqrt(1 - e2 * np.sin(np.radians(phi)) ** 2)
    X = lambda phi, lamb: N_phi(phi) * cosd(phi) * cosd(lamb)
    Y = lambda phi, lamb: N_phi(phi) * cosd(phi) * sind(lamb)
    Z = lambda phi, alt: (b ** 2 / a ** 2 * N_phi(phi) + alt) * sind(phi)

    ecef = np.zeros(coords.shape)
    if len(ecef.shape) > 1:
        for ij in range(coords.shape[0]):
            p = coords[ij, 0]
            l = coords[ij, 1]
            h = coords[ij, 2]
            ecef[ij, :] = np.array([X(p, l), Y(p, l), Z(p, h)])
    else:
        p = coords[0]
        l = coords[1]
        h = coords[2]
        ecef = np.array([X(p, l), Y(p, l), Z(p, h)])

    return ecef


def ECEF_to_geodetic(ecef_coords: np.ndarray) -> np.ndarray:
    a, b, e2 = ecef_constants()
    geodetic_coords = np.zeros(ecef_coords.shape)

    X = ecef_coords[:, 0]
    Y = ecef_coords[:, 1]
    Z = ecef_coords[:, 2]

    r2 = np.square(X) + np.square(Y)
    r = np.sqrt(r2)
    Z2 = np.square(Z)
    er2 = (a**2 - b**2)/b**2
    F = 54*b**2 * Z2
    G = r2 + (1-e2)*Z2-e2*(a**2-b**2)
    c = e2**2*F*r2 / np.power(G, 3)
    s = np.power(1+c+np.sqrt(np.square(c) + 2*c), 1/3)
    P = F / (3*np.square(s+1+1/s)*np.square (G))
    Q = np.sqrt(1+2*e2**2*P)
    r0 = -P*e2*r / (1+Q) + np.sqrt(1/2*a**2 * (1+1/Q) - P*(1-e2)*Z2 / (Q*(1+Q)) - 1/2*P*r2)
    U = np.sqrt(np.square(r-e2*r0) + Z2)
    V = np.sqrt(np.square(r-e2*r0) + (1-e2)*Z2)
    z0 = b**2 * Z / (a*V)
    h = U*(1-b**2 / (a*V))
    phi = np.rad2deg(np.arctan((Z+er2*z0) / r))
    lam = np.rad2deg(np.arctan2(Y, X))

    return np.transpose(np.array([lam, phi, h]))


def reference_point() -> Tuple[np.array, np.array, np.array]:
    # Reference point is Reser Bicycle Outfitters
    ref_lat = 39.09029667468314  # deg
    ref_lon = -84.49260971579635  # deg
    ref_alt = 156.058  # m
    ref_ecef = geodetic_to_ECEF(np.array([ref_lat, ref_lon, ref_alt]))

    mat_transform = np.array([[-sind(ref_lon), cosd(ref_lon), 0],
                              [-sind(ref_lat) * cosd(ref_lon), -sind(ref_lat) * sind(ref_lon), cosd(ref_lat)],
                              [cosd(ref_lat) * cosd(ref_lon), cosd(ref_lat) * sind(ref_lon), sind(ref_lat)]])

    return np.array([ref_lat, ref_lon, ref_alt]), ref_ecef, mat_transform


def ECEF_to_ENU(coords: np.ndarray) -> np.ndarray:
    # Construct the matrix
    ref_geo, ref_ecef, mat_transform = reference_point()
    enu_coords = np.zeros(coords.shape)
    for ij in range(coords.shape[0]):
        enu_coords[ij, :] = np.dot(mat_transform, coords[ij, :] - ref_ecef)

    return enu_coords


def ENU_to_ECEF(enu_coords: np.array) -> np.array:
    # Construct the matrix
    ref_geo, ref_ecef, mat_transform = reference_point()
    # Transpose to go the other way
    mat_transform = np.transpose(mat_transform)
    coords = np.zeros(enu_coords.shape)
    for ij in range(enu_coords.shape[0]):
        coords[ij, :] = np.dot(mat_transform, enu_coords[ij, :]) + ref_ecef

    return coords


def is_report(placemark) -> bool:
    return get_extended_data(placemark.ExtendedData).get('official', '').upper() == 'REPORT'


def parse_file(file_name: str):
    with open(file_name) as f:
        return pykml.parser.parse(f).getroot()


def geo_coords(s: str) -> np.ndarray:
    # Lat, Lon, h
    return np.array(tuple(map(float, s.split(','))))[[1, 0, 2]]


def split_string(x: str) -> List[str]:
    return [line.strip() for line in x.split('\n') if line.strip()]


def get_extended_data(x) -> Dict[str, str]:
    return dict([(d.attrib.values()[0], csv_format_string((d.value.text or "").strip())) for d in x.Data])


def csv_format_string(x: str) -> str:
    return x.replace('\n', ' ').replace('\r', ' ').replace('\f', ' ').replace(',', '')


def sind(x: float) -> float:
    return np.sin(np.radians(x))


def cosd(x: float) -> float:
    return np.cos(np.radians(x))
