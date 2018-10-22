from flopy.utils.flopy_io import get_url_text
import numpy as np

try:
    import pandas as pd
except:
    pd = False


class acdd:
    """
    Translate ScienceBase global metadata attributes to CF and ACDD
    global attributes.

    Parameters
    ----------

    sciencebase_id : str
        Unique identifier for ScienceBase record (e.g. 582da7efe4b04d580bd37be8)
    model : flopy model object
        Model object

    References
    ----------

    https://www.sciencebase.gov/catalog/
    http://cfconventions.org/cf-conventions/v1.6.0/cf-conventions.html#description-of-file-contents
    http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery

    """

    def __init__(self, sciencebase_id, model):
        """
        Class constructor
        """

        self.id = sciencebase_id
        self.model = model
        self.sciencebase_url = 'https://www.sciencebase.gov/catalog/item/{}'.format(
            sciencebase_id)
        self.sb = self.get_sciencebase_metadata(sciencebase_id)
        if self.sb is None:
            return

        # stuff Jeremy mentioned
        self.abstract = self.sb['summary']
        self.authors = [c['name'] for c in self.sb['contacts']
                        if 'Originator' in c['type']]
        # report image?

        # keys that are the same in sbjson and acdd;
        # or additional attributes to carry over
        for k in ['title', 'summary', 'id', 'citation']:
            self.__dict__[k] = self.sb.get(k, None)

        # highly recommended global attributes
        # http://wiki.esipfed.org/index.php/Attribute_Convention_for_Data_Discovery
        self.keywords = [t['name'] for t in self.sb['tags']]

        # recommended global attributes
        self.naming_authority = 'ScienceBase'  # org. that provides the id
        # self.history = None # This is a character array with a line for each invocation of a program that has modified the dataset.
        # Well-behaved generic netCDF applications should append a line containing:
        # date, time of day, user name, program name and command arguments.
        self.source = model.model_ws  # The method of production of the original data.
        # If it was model-generated, source should name the model and its version.
        # self.processing_level = None # 	A textual description of the processing (or quality control) level of the data.
        # self.comment = None #	Miscellaneous information about the data, not captured elsewhere.
        # This attribute is defined in the CF Conventions.
        self.acknowledgement = self._get_xml_attribute('datacred')
        # self.license = None #
        # self.standard_name_vocabulary = None
        self.date_created = self.sb['provenance']['linkProcess'].get(
            'dateCreated')
        self.creator_name = self.creator.get('name')
        self.creator_email = self.creator.get('email')
        # self.creator_url = self.sb['webLinks'][0].get('uri')
        self.creator_institution = self.creator['organization'].get(
            'displayText')
        self.institution = self.creator_institution  # also in CF convention for global attributes
        self.project = self.sb['title']
        self.publisher_name = [d.get('name') for d in self.sb['contacts'] if
                               'publisher' in d.get('type').lower()][0]
        self.publisher_email = self.sb['provenance']['linkProcess'].get(
            'processedBy')
        self.publisher_url = 'https://www2.usgs.gov/water/'  # self.sb['provenance']['linkProcess'].get('linkReference')
        self.geospatial_bounds_crs = 'EPSG:4326'
        self.geospatial_lat_min = self.bounds.get('minY')
        self.geospatial_lat_max = self.bounds.get('maxY')
        self.geospatial_lon_min = self.bounds.get('minX')
        self.geospatial_lon_max = self.bounds.get('maxX')
        self.geospatial_vertical_min = self.model.dis.botm.array.min()
        self.geospatial_vertical_max = self.model.dis.top.array.max()
        self.geospatial_vertical_positive = 'up'  # assumed to always be up for GW models
        self.time_coverage_start = self.time_coverage.get('start')
        self.time_coverage_end = self.time_coverage.get('end')
        self.time_coverage_duration = self.time_coverage.get('duration')
        # because the start/end date formats aren't consistent between models
        self.time_coverage_resolution = self.time_coverage.get('resolution')

        self.metadata_link = self.sciencebase_url

    def _get_xml_attribute(self, attr):
        try:
            return list(self.xmlroot.iter(attr))[0].text
        except:
            return None

    @property
    def bounds(self):
        return self.sb['spatial']['boundingBox']

    @property
    def creator(self):
        return [d for d in self.sb['contacts'] if
                'point of contact' in d['type'].lower()][0]

    @property
    def creator_url(self):
        urlname = '-'.join(self.creator.get('name').replace('.', '').split())
        url = 'https://www.usgs.gov/staff-profiles/' + urlname.lower()
        # check if it exists
        txt = get_url_text(url)
        if txt is not None:
            return url
        else:
            return 'unknown'

    @property
    def geospatial_bounds(self):
        """
        Describes the data's 2D or 3D geospatial extent in OGC's Well-Known
        Text (WKT) Geometry format
        """
        fmt = '(({0} {2}, {0} {3}, {1} {3}, {1} {2}, {0} {2}))'
        bounds = 'POLYGON ' + fmt.format(self.geospatial_lon_min,
                                         self.geospatial_lon_max,
                                         self.geospatial_lat_min,
                                         self.geospatial_lat_max)
        return bounds

    @property
    def geospatial_bounds_vertical_crs(self):
        """
        The vertical coordinate reference system (CRS) for the Z axis of
        the point coordinates in the geospatial_bounds attribute.
        """
        epsg = {'NGVD29': 'EPSG:5702', 'NAVD88': 'EPSG:5703'}
        return epsg.get(self.vertical_datum)

    @property
    def references(self):
        """

        Returns
        -------

        """
        r = [self.citation]
        links = [d.get('uri') for d in self.sb['webLinks']
                 if 'link' in d.get('type').lower()]
        return r + links

    @property
    def time_coverage(self):
        """

        Returns
        -------

        """
        l = self.sb['dates']
        tc = {}
        for t in ['start', 'end']:
            tc[t] = [d.get('dateString') for d in l
                     if t in d['type'].lower()][0]
        if not np.all(self.model.dis.steady) and pd:
            # replace with times from model reference
            tc['start'] = self.model.dis.start_datetime
            strt = pd.Timestamp(self.model.dis.start_datetime)
            mlen = self.model.dis.perlen.array.sum()
            tunits = self.model.dis.itmuni_dict[self.model.dis.itmuni]
            tc['duration'] = '{} {}'.format(mlen, tunits)
            end = strt + pd.Timedelta(mlen, unit='d')
            tc['end'] = str(end)
        return tc

    @property
    def vertical_datum(self):
        """
        Try to parse the vertical datum from the xml info
        """
        altdatum = self._get_xml_attribute('altdatum')
        if altdatum is not None:
            if '88' in altdatum:
                return 'NAVD88'
            elif '29' in altdatum:
                return 'NGVD29'
        else:
            return None

    @property
    def xmlroot(self):
        """
        ElementTree root element object for xml metadata
        """
        try:
            return self.get_sciencebase_xml_metadata()
        except:
            None

    @property
    def xmlfile(self):
        return self.sb['identifiers'][0].get('key')

    def get_sciencebase_metadata(self, id):
        """
        Gets metadata json text for given ID from sciencebase.gov; loads
        into python dictionary. Fetches the reference text using the url:
        https://www.sciencebase.gov/catalog/item/<ID>?format=json

        Parameters
        ----------
        ID : str
            ScienceBase ID string; 
            e.g. 582da7efe4b04d580bd37be8 for Dane County Model

        Returns
        -------
        metadata : dict
            Dictionary of metadata
        """
        urlbase = 'https://www.sciencebase.gov/catalog/item/{}?format=json'
        url = urlbase.format(id)

        import json
        from flopy.utils.flopy_io import get_url_text
        msg = 'Need an internet connection to get metadata from ScienceBase.'
        text = get_url_text(url, error_msg=msg)
        if text is not None:
            return json.loads(text)

    def get_sciencebase_xml_metadata(self):
        """
        Gets xml from sciencebase.gov, using XML url obtained
        from json using get_sciencebase_metadata().

        Parameters
        ----------
        ID : str
            ScienceBase ID string; 
            e.g. 582da7efe4b04d580bd37be8 for Dane County Model

        Returns
        -------
        metadata : dict
            Dictionary of metadata
        """
        import xml.etree.ElementTree as ET
        from flopy.utils.flopy_io import get_url_text

        url = self.xmlfile
        msg = 'Need an internet connection to get metadata from ScienceBase.'
        text = get_url_text(url, error_msg=msg)
        return ET.fromstring(text)
