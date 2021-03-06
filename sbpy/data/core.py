# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
================
sbpy Data Module
================

created on June 22, 2017
"""

from collections import OrderedDict
from numpy import ndarray, array
from astropy.table import QTable, Column, vstack
import astropy.units as u

from . import conf

__all__ = ['DataClass', 'mpc_observations', 'sb_search', 'image_search',
           'pds_ferret']


class DataClass():
    """`~sbpy.data.DataClass` serves as the base class for all data
    container classes in `sbpy` in order to provide consistent
    functionality throughout all these classes.

    The core of `~sbpy.data.DataClass` is an `~astropy.table.QTable`
    object (referred to as the `data table` below) - a type of
    `~astropy.table.Table` object that supports the `~astropy.units`
    formalism on a per-column base - which already provides most of
    the required functionality. `~sbpy.data.DataClass` objects can be
    manually generated from dictionaries
    (`~sbpy.data.DataClass.from_dict`), `~numpy.array`-like
    (`~sbpy.data.DataClass.from_array`) objects, or directly from
    another `~astropy.table.QTable` object.

    A few high-level functions for table data access or modification
    are provided; other, more complex modifications can be applied to
    the underlying table object (`~sbpy.data.DataClass.table`) directly.

    """

    def __init__(self, data):
        self._table = QTable()
        # self.altkeys = {}  # dictionary for alternative column names

        if (len(data.items()) == 1 and 'table' in data.keys()):
            # single item provided named 'table' -> already Table object
            self._table = QTable(data['table'])
        else:
            # treat kwargs as dictionary
            for key, val in data.items():
                try:
                    unit = val.unit
                    val = val.value
                except AttributeError:
                    unit = None

                # check if val is already list-like
                try:
                    val[0]
                except TypeError:
                    val = [val]

                self._table[key] = Column(val, unit=unit)

    @classmethod
    def from_dict(cls, data):
        """Create `~sbpy.data.DataClass` object from dictionary or list of
        dictionaries.

        Parameters
        ----------
        data : `~collections.OrderedDict`, dictionary or list (or similar) of
             dictionaries Data that will be ingested in
             `~sbpy.data.DataClass` object.  Each dictionary creates a
             row in the data table. Dictionary keys are used as column
             names; corresponding values must be scalar (cannot be
             lists or arrays). If a list of dictionaries is provided,
             all dictionaries have to provide the same set of keys
             (and units, if used at all).

        Returns
        -------
        `DataClass` object

        Examples
        --------
        >>> import astropy.units as u
        >>> from sbpy.data import Orbit
        >>> orb = Orbit.from_dict({'a': 2.7674*u.au,
        ...                        'e': 0.0756,
        ...                        'i': 10.59321*u.deg})

        Since dictionaries have no specific order, the ordering of the
        column in the example above is not defined. If your data table
        requires a specific order, use an ``OrderedDict``:

        >>> from collections import OrderedDict
        >>> orb = Orbit.from_dict(OrderedDict([('a', 2.7674*u.au),
        ...                                    ('e', 0.0756),
        ...                                    ('i', 10.59321*u.deg)]))
        >>> print(orb)
        <sbpy.data.orbit.Orbit object at 0x...>
        >>> print(orb.column_names) # doctest: +SKIP
        <TableColumns names=('a','e','i')>
        >>> print(orb.table['a', 'e', 'i'])
          a      e       i
          AU            deg
        ------ ------ --------
        2.7674 0.0756 10.59321

        """
        if isinstance(data, (dict, OrderedDict)):
            return cls(data)
        elif isinstance(data, (list, ndarray, tuple)):
            # build table from first dict and append remaining rows
            tab = cls(data[0])
            for row in data[1:]:
                tab.add_rows(row)
            return tab
        else:
            raise TypeError('this function requires a dictionary or a '
                            'list of dictionaries')

    @classmethod
    def from_array(cls, data, names):
        """Create `~sbpy.data.DataClass` object from list, `~numpy.ndarray`,
        or tuple.

        Parameters
        ----------
        data : list, `~numpy.ndarray`, or tuple
            Data that will be ingested in `DataClass` object. A one
            dimensional sequence will be interpreted as a single row. Each
            element that is itself a sequence will be interpreted as a
            column.
        names : list
            Column names, must have the same number of names as data columns.

        Returns
        -------
        `DataClass` object

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.deg,
        ...                             [4, 5, 6]*u.km,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> print(dat.table)
         a   b   c
        deg  km
        --- --- ---
        1.0 4.0   a
        2.0 5.0   b
        3.0 6.0   c

        """

        if isinstance(data, (list, ndarray, tuple)):
            return cls.from_dict(OrderedDict(zip(names, data)))
        else:
            raise TypeError('this function requires a list, tuple or a '
                            'numpy array')

    @classmethod
    def from_table(cls, data):
        """Create `DataClass` object from `~astropy.table.Table` or
        `astropy.table.QTable` object.

        Parameters
        ----------
        data : astropy `Table` object, mandatory
             Data that will be ingested in `DataClass` object.

        Returns
        -------
        `DataClass` object

        Examples
        --------
        >>> from astropy.table import QTable
        >>> import astropy.units as u
        >>> from sbpy.data import DataClass
        >>> tab = QTable([[1,2,3]*u.kg,
        ...               [4,5,6]*u.m/u.s,],
        ...              names=['mass', 'velocity'])
        >>> dat = DataClass.from_table(tab)
        >>> print(dat.table)
        mass velocity
         kg   m / s
        ---- --------
         1.0      4.0
         2.0      5.0
         3.0      6.0
        """

        return cls({'table': data})

    @classmethod
    def from_file(cls, filename, **kwargs):
        """Create `DataClass` object from a file using
        `~astropy.table.Table.read`.

        Parameters
        ----------
        filename : str
             Name of the file that will be read and parsed.
        **kwargs : additional parameters
             Optional parameters that will be passed on to
             `~astropy.table.Table.read`.

        Returns
        -------
        `DataClass` object

        Notes
        -----
        This function is merely a wrapper around
        `~astropy.table.Table.read`. Please refer to the documentation of
        that function for additional information on optional parameters
        and data formats that are available. Furthermore, note that this
        function is not able to identify units. If you want to work with
        `~astropy.units` you have to assign them manually to the object
        columns.

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> dat = DataClass.from_file('data.txt', format='ascii')  # doctest: +SKIP
        """

        data = QTable.read(filename, **kwargs)

        return cls({'table': data})

    def to_file(self, filename, format='ascii', **kwargs):
        """Write object to a file using
        `~astropy.table.Table.write`.

        Parameters
        ----------
        filename : str
             Name of the file that will be written.
        format : str, optional
             Data format in which the file should be written. Default:
             ``ASCII``
        **kwargs : additional parameters
             Optional parameters that will be passed on to
             `~astropy.table.Table.write`.

        Returns
        -------
        None

        Notes
        -----
        This function is merely a wrapper around
        `~astropy.table.Table.write`. Please refer to the
        documentation of that function for additional information on
        optional parameters and data formats that are
        available. Furthermore, note that this function is not able to
        write unit information to the file.

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.deg,
        ...                             [4, 5, 6]*u.km,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> dat.to_file('test.txt')

        """

        self._table.write(filename, format=format, **kwargs)

    def __getattr__(self, field):
        """Get attribute from ``self._table` (columns, rows); checks
        for and may use alternative field names."""

        if field in dir(self):
            return self.field
        else:
            if len(self._translate_columns(field)) > 0:
                field = self._translate_columns(field)[0]
                return self._table[field]
            else:
                raise AttributeError('Attribute {:s} not available.'.format(
                    field))

    def __setattr__(self, field, value):
        """Modify attribute in ``self._table``, if it already exists there,
        or set it in ``self``."""
        try:
            # if self._table exists ...
            if field in self._table.columns:
                # set value there...
                self._table[field] = value
            else:
                super().__setattr__(field, value)
        except:
            # if, not set it for self
            super().__setattr__(field, value)

    def __getitem__(self, ident):
        """Return columns or rows from data table (``self._table``); checks
        for and may use alternative field names."""

        if isinstance(ident, str):
            if len(self._translate_columns(ident)) > 0:
                ident = self._translate_columns(ident)[0]

        return self._table[ident]

    @property
    def table(self):
        """Return `~astropy.table.QTable` object containing all data."""
        return self._table

    @property
    def column_names(self):
        """Return a list of all column names in the data table."""
        return self._table.columns

    def add_rows(self, rows):
        """Append additional rows to the existing data table. An individual
        row can be provided in list, tuple, `~numpy.ndarray`, or
        dictionary form. Multiple rows can be provided in the form of
        a list, tuple, or `~numpy.ndarray` of individual
        rows. Multiple rows can also be provided in the form of a
        `~astropy.table.QTable` or another `~sbpy.data.DataClass`
        object. In case of a dictionary, `~astropy.table.QTable`, or
        `~sbpy.data.DataClass`, all table column names must be
        provided in ``row``; additional keys that are not yet column
        names in the table will be discarded. In case of a list, the
        list elements must be in the same order as the table
        columns. In either case, matching `~astropy.units` must be
        provided in ``rows`` if used in the data table.

        Parameters
        ----------
        rows : list, tuple, `~numpy.ndarray`, dict, or `~collections.OrderedDict`
            data to be appended to the table; required to have the same
            length as the existing table, as well as the same units

        Returns
        -------
        n : int, the total number of rows in the data table

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.Unit('m'),
        ...                             [4, 5, 6]*u.m/u.s,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> dat.add_rows({'a': 5*u.m, 'b': 8*u.m/u.s, 'c': 'e'})
        4
        >>> print(dat.table)
         a    b    c
         m  m / s
        --- ----- ---
        1.0   4.0   a
        2.0   5.0   b
        3.0   6.0   c
        5.0   8.0   e
        >>> dat.add_rows(([6*u.m, 9*u.m/u.s, 'f'],
        ...               [7*u.m, 10*u.m/u.s, 'g']))
        6
        >>> dat.add_rows(dat)
        12
        """
        if isinstance(rows, QTable):
            self._table = vstack([self._table, rows])
        if isinstance(rows, DataClass):
            self._table = vstack([self._table, rows.table])
        if isinstance(rows, (dict, OrderedDict)):
            try:
                newrow = [rows[colname] for colname in self._table.columns]
            except KeyError as e:
                raise ValueError('data for column {0} missing in row {1}'.
                                 format(e, rows))
            self.add_rows(newrow)
        if isinstance(rows, (list, ndarray, tuple)):
            if (not isinstance(rows[0], (u.quantity.Quantity, float)) and
                    isinstance(rows[0], (dict, OrderedDict,
                                         list, ndarray, tuple))):
                for subrow in rows:
                    self.add_rows(subrow)
            else:
                self._table.add_row(rows)
        return len(self._table)

    def add_column(self, data, name):
        """Append a single column to the current data table. The lenght of
        the input list, `~numpy.ndarray`, or tuple must match the current
        number of rows in the data table.

        Parameters
        ----------
        data : list, `~numpy.ndarray`, or tuple
            Data to be filled into the table; required to have the same
            length as the existing table's number rows.
        name : str
            Name of the new column; must be different from already existing
            column names.

        Returns
        -------
        n : int, the total number of columns in the data table

        Examples
        --------
        >>> from sbpy.data import DataClass
        >>> import astropy.units as u
        >>> dat = DataClass.from_array([[1, 2, 3]*u.Unit('m'),
        ...                             [4, 5, 6]*u.m/u.s,
        ...                             ['a', 'b', 'c']],
        ...                            names=('a', 'b', 'c'))
        >>> dat.add_column([10, 20, 30]*u.kg, name='d')
        4
        >>> print(dat.table)
         a    b    c   d
         m  m / s      kg
        --- ----- --- ----
        1.0   4.0   a 10.0
        2.0   5.0   b 20.0
        3.0   6.0   c 30.0
        """

        self._table.add_column(Column(data, name=name))
        return len(self.column_names)

    def _translate_columns(self, target_colnames):
        """Translate target_colnames to the corresponding column names
        present in this object's table. Returns a list of actual column
        names present in this object that corresponds to target_colnames
        (order is preserved). Raises ValueError if not all columns are
        present or one or more columns could not be translated
        """

        if not isinstance(target_colnames, (list, ndarray, tuple)):
            target_colnames = [target_colnames]

        translated_colnames = []
        for colname in target_colnames:
            if colname in self._table.columns:
                # colname already in self._table
                translated_colnames.append(colname)
            elif colname in conf.namealts.keys():
                # colname already default name (conf.namealts key)
                for translation in conf.namealts[colname]:
                    if translation in self.column_names:
                        translated_colnames.append(translation)
            else:
                # colname is alternative name (conf.altnames key)
                if colname in conf.altnames.keys():
                    translated_colnames.append(conf.altnames[colname])
                else:
                    raise KeyError('column \"{:s}\" not available'.format(
                        colname))

        return translated_colnames


def mpc_observations(targetid):
    """Obtain all available observations of a small body from the Minor
    Planet Center (http://www.minorplanetcenter.net) and provides them in
    the form of an Astropy table.

    Parameters
    ----------
    targetid : str, mandatory
        target identifier

    Returns
    -------
    `~sbpy.data.DataClass` object

    Examples
    --------
    >>> from sbpy.data import mpc_observations
    >>> obs = mpc_observations('ceres')  # doctest: +SKIP

    not yet implemented

    """


def sb_search(field):
    """Use the Skybot service (http://vo.imcce.fr/webservices/skybot/) at
    IMCCE to Identify moving objects potentially present in a registered
    FITS images.

    Parameters
    ----------
    field : string, astropy.io.fits Header object, Primary HDU, or Image HDU
      A FITS image file name, HDU data structure, or header with
      defined WCS

    Returns
    -------
    `~sbpy.data.DataClass` object

    Examples
    --------
    >>> from sbpy.data import sb_search
    >>> objects = sb_search('ceres')  # doctest: +SKIP

    not yet implemented

    """


def image_search(targetid):
    """Use the Solar System Object Image Search function of the Canadian
    Astronomy Data Centre
    (http://www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca/en/ssois/) to identify
    images with a specific small body in them.

    Parameters
    ----------
    targetid : str, mandatory
        target identifier

    Returns
    -------
    `~sbpy.data.DataClass` object

    Examples
    --------
    >>> from sbpy.data import Misc  # doctest: +SKIP
    >>> images = Misc.image_search('ceres')  # doctest: +SKIP

    not yet implemented

    """


def pds_ferret(targetid):
    """Use the Small Bodies Data Ferret (http://sbntools.psi.edu/ferret/)
    at the Planetary Data System's Small Bodies Node to query for
    information on a specific small body in the PDS.

    Parameters
    ----------
    targetid : str, mandatory
        target identifier

    Returns
    -------
    data : dict
      A hierarchical data object

    Examples
    --------
    >>> from sbpy.data import pds_ferret
    >>> data = pds_ferret('ceres')  # doctest: +SKIP

    not yet implemented

    """
