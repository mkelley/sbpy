Data Module (`sbpy.data`)
=========================

Introduction
------------

`sbpy.data` provides classes for dealing with orbital elements
(`~sbpy.data.Orbit`), ephemerides (`~sbpy.data.Ephem`), and physical
properties (`~sbpy.data.Phys`). `~sbpy.data.Ephem`,
`~sbpy.data.Orbit`, and `~sbpy.data.Phys` objects act as containers
for such parameters and can (and should) be used to provide these to
functions in `sbpy`. Each of these classes is based on the
`~sbpy.data.DataClass` base class, which internally uses an
`~astropy.table.QTable` object and provides the same functionality and
features as the latter.

Furthermore, `~sbpy.data` also provides additional interfaces to a number of
different services and `~sbpy.data.Names` provides functions
related to naming conventions for asteroids and comets.


How to use Ephem, Orbit, and Phys objects
-----------------------------------------

All of the data objects dealt with in `sbpy.data` share the same
common base class: `sbpy.data.DataClass`. `~sbpy.data.DataClass`
defines the basic functionality and makes sure that all `sbpy.data`
objects can used in the exact same way.

In plain words, this means that in the following examples you can
replace `~sbpy.data.DataClass`, `~sbpy.data.Ephem`,
`~sbpy.data.Orbit`, and `~sbpy.data.Phys` object with each other. In
order to show some useful use cases, we will iterate between these
types, but keep in mind: they all work the exact same way.

`~sbpy.data.DataClass` uses `~astropy.table.QTable` objects under the
hood. You can think of those as tables - consisting of columns and
rows - that have `~astropy.units` attached to them, allowing you to
propagate these units through your code. Each `~sbpy.data.DataClass`
object can hold as many data as you want, where each datum can be a
different object or the same object at a different epoch.


Building an object
^^^^^^^^^^^^^^^^^^

While `~sbpy.data.Ephem`, `~sbpy.data.Orbit`, and `~sbpy.data.Phys`
provide a range of convience functions to build objects containing
data, for instance from online data archives, it is easily possible to
build these objects from scratch. This can be done for input data
stored in dictionaries (`~sbpy.data.DataClass.from_dict`), lists or
arrays (`~sbpy.data.DataClass.from_array`), `~astropy.table.Table`
objects (`~sbpy.data.DataClass.from_table`), or from data files
(`~sbpy.data.DataClass.from_file`).

Depending on how your input data are organized, you can use different
options in different cases:

1. Assume that you want to build a `~sbpy.data.Orbit` object to
   propagate this orbit and obtain ephemerides. Since you are dealing
   with a single orbit, the most convenient solution might be to use a
   dictionary to build your object:

    >>> from sbpy.data import Orbit
    >>> import astropy.units as u
    >>> elements = {'a':1.234*u.au, 'e':0.1234, 'i':12.34*u.deg,
    ...             'argper': 123.4*u.deg, 'node': 45.2*u.deg,
    ...             'epoch': 2451200.5*u.d, 'true_anom':23.1*u.deg}
    >>> orb = Orbit.from_dict(elements)
    >>> print(orb)  # doctest:+ELLIPSIS
    <sbpy.data.orbit.Orbit object at ...>

   One quick note on building `~sbpy.data.DataClass` objects from
   dictionaries: dictionaries have no intrinsic order. In dictionary
   ``elements`` as defined here, there is no guarantee that ``'a'``
   will always be located before ``'e'`` when reading out the
   dictionary item by item, which happens when the data table is built
   in the background. Hence, the order of the resulting data table
   columns has to be considered random. If you want to force a
   specific order on the columns in your data table, you can use and
   `~collections.OrderedDict` instead of a simple dictionary. The
   order of elements in an `~collections.OrderedDict` will be the same
   as the order of the data table columns.

2. Now assume that you want to build an `~sbpy.data.Ephem` object
   holding RA, Dec, and observation midtime for some target that you
   observed. In this case, you could provide a list of three
   dictionaries to `~sbpy.data.DataClass.from_dict`, which means a lot
   of typing. Instead, you can use `~sbpy.data.DataClass.from_array`,
   which allows to provide your input data in the form of a list,
   tuple, or `~numpy.ndarray`:

    >>> from sbpy.data import Ephem
    >>> import astropy.units as u
    >>> from numpy import array
    >>> ra = [10.223423, 10.233453, 10.243452]*u.deg
    >>> dec = [-12.42123, -12.41562, -12.40435]*u.deg
    >>> epoch = (2451523.5 + array([0.1234, 0.2345, 0.3525]))*u.d
    >>> obs = Ephem.from_array([ra, dec, epoch], names=['ra', 'dec', 't'])
    >>> print(obs)  # doctest:+ELLIPSIS   
    <sbpy.data.ephem.Ephem object at ...>

3. If your data are already available as a `~astropy.table.Table` or
   `~astropy.table.QTable`, you can simply convert it into a
   `~sbpy.data.DataClass` object using
   `~sbpy.data.DataClass.from_table`.

4. You can also read in the data from a file that should be properly
   formatted (e.g., it should have a headline with the same number of
   elements as there are columns) using
   `~sbpy.data.DataClass.from_file`. This function merely serves as a
   wrapper for `~astropy.table.Table.read` and uses the same
   parameters as the latter function. You can read in an ASCII file
   using the following lines:

   >>> from sbpy.data import Ephem
   >>> data = Ephem.from_file('data.txt', format='ascii') # doctest: +SKIP

   Please not that `~sbpy.data.DataClass.from_file` is not able to
   identify units automatically. If you want to take advantage for
   `~astropy.units` you will have to assign these units manually later
   on.


Accessing data
^^^^^^^^^^^^^^

In order to obtain a list of column names in a `~sbpy.data.DataClass` object, you can use `~sbpy.data.DataClass.column_names`:

    >>> obs.column_names
    <TableColumns names=('ra','dec','t')>

Each of these columns can be accessed easily, for instance:

    >>> print(obs['ra']) # doctest: +SKIP
    [10.223423 10.233453 10.243452] deg

Similarly, if you are interested in the first set of observations in
``obs``, you can use:

    >>> print(obs[0])
        ra       dec         t      
       deg       deg         d      
    --------- --------- ------------
    10.223423 -12.42123 2451523.6234

which returns you a table with only the requested subset of the
data. In order to retrieve RA from the second observation, you can
combine both examples and do:

    >>> print(obs[1]['ra'])
    10.233453 deg

Another - maybe more elegant - way to access data table columns is as
an attribute:

    >>> print(obs.ra) # doctest: +SKIP
    [10.223423 10.233453 10.243452] deg
    >>> print(obs.ra[1])
    10.233453 deg
    
Just like in any `~astropy.table.Table` or `~astropy.table.QTable` object, you can use slicing to obtain subset tables from your data, for instance:

    >>> print(obs['ra', 'dec'])
        ra       dec   
       deg       deg   
    --------- ---------
    10.223423 -12.42123
    10.233453 -12.41562
    10.243452 -12.40435

    >>> print(obs[obs['ra'] <= 10.233453*u.deg])
        ra       dec         t      
       deg       deg         d      
    --------- --------- ------------
    10.223423 -12.42123 2451523.6234
    10.233453 -12.41562 2451523.7345

The latter uses a condition to filter data (only those observations
with RA less than or equal to 10.233453 degrees; note that it is
necessary here to apply ``u.deg`` to the value that all the RAs are
compared against) but selects all the columns in the original table.

If you ever need to access the actual `~astropy.table.QTable` object
that is inside each `~sbpy.data.DataClass` object, you can access it
as ``obs.table``.
    
Modifying an object
^^^^^^^^^^^^^^^^^^^

`~sbpy.data.DataClass` offers some convenience functions for object
modifications. It is trivial to add additional rows and columns to
these objects in the form of lists, arrays, or dictionaries. 

Let's assume you want to add some more observations to your ``obs``
object:

    >>> obs.add_rows([[10.255460*u.deg, -12.39460*u.deg, 2451523.94653*u.d],
    ...               [10.265425*u.deg, -12.38246*u.deg, 2451524.0673*u.d]])
    5
    >>> print(obs.table)
        ra       dec          t      
       deg       deg          d      
    --------- --------- -------------
    10.223423 -12.42123  2451523.6234
    10.233453 -12.41562  2451523.7345
    10.243452 -12.40435  2451523.8525
     10.25546  -12.3946 2451523.94653
    10.265425 -12.38246  2451524.0673

or if you want to add a column to your object:

    >>> obs.add_column(['V', 'V', 'R', 'i', 'g'], name='filter')
    4
    >>> print(obs.table)
        ra       dec          t       filter
       deg       deg          d             
    --------- --------- ------------- ------
    10.223423 -12.42123  2451523.6234      V
    10.233453 -12.41562  2451523.7345      V
    10.243452 -12.40435  2451523.8525      R
     10.25546  -12.3946 2451523.94653      i
    10.265425 -12.38246  2451524.0673      g
    
A few things to be mentioned here:

* Note how both functions return the number of rows or columns in the
  updated object.
* If you are adding rows, the elements in the rows will be assigned to
  the column in the corresponding order of the table columns. The
  `~astropy.units` of the row elements have to be of the same
  dimension as the table columns (e.g., one of the table column units
  is degrees, then the corresponding row element has to define an
  angular distance: ``u.deg`` or ``u.rad``).
* Naturally, the number of columns and rows of the rows and columns
  to be added has to be identical to the numbers in the data table.

If you are trying to add a single row to your object data table, using a dictionary might be the most convenient solution:

    >>> obs.add_rows({'ra':10.255460*u.deg, 'dec': -12.39460*u.deg,
    ...               't': 2451524.14653*u.d, 'filter': 'z'})
    6

    
When adding a large number of rows to your object, it might be most
convenient to first convert all the new rows into new
`~sbpy.data.DataClass` object and then append that using
`~sbpy.data.DataClass.add_rows`:

    >>> obs2 = Ephem.from_array([[10.4545, 10.5656]*u.deg,
    ...                          [-12.1212, -12.0434]*u.deg,
    ...                          [2451524.14653, 2451524.23541]*u.d,
    ...                          ['r', 'z']],
    ...                         names=['ra', 'dec', 't', 'filter'])
    >>> obs.add_rows(obs2)
    8

Individual elements, entire rows, and columns can be modified by
directly addressing them:

    >>> print(obs['ra']) # doctest: +SKIP
    [10.223423 10.233453 10.243452 10.25546  10.265425 10.25546  10.4545
     10.5656  ] deg
    >>> obs['ra'][:] = obs['ra'] + 0.1*u.deg
    >>> print(obs['ra']) # doctest: +SKIP
    [10.323423 10.333453 10.343452 10.35546  10.365425 10.35546  10.5545
     10.6656  ] deg

Note the specific syntax in this case (``obs['ra'][:] = ...``) that
is required by `~astropy.table.Table` if you want to replace
an entire column.
    
More complex data table modifications are possible by directly
accessing the underlying `~astropy.table.QTable` object.

Writing object data to a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`~sbpy.data.DataClass` objects can be written to files using
`~sbpy.data.DataClass.to_file`:

    >>> obs.to_file('observations.dat')

By default, the data are written in ASCII format, but other formats
are available, too (cf. `~astropy.table.Table.write`).

Alternative field names
^^^^^^^^^^^^^^^^^^^^^^^

It is common practice to use a set of different names for the same
property. For instance, the orbital inclination can be referred to as
``'i'``, ``'inc'``, or ``'incl'`` - it's a matter of personal
taste. `~sbpy.data.DataClass` accounts for this fact and is able to
provide a number of alternative field or property names, as suggested
above.

As an example, if your `~sbpy.data.Orbit` object has a column named
``'incl'`` but you try to get column ``'i'``, the object will
internally check if ``'i'`` is a legitimate alternative field name for
``'incl'``. The corresponding column is then returned. If you try to
get a field name that is not connected to any existing field name, a
``KeyError`` will be raised.

The definition of alternative field names is done in the file
``sbpy/data/__init__.py``, using the dictionary ``namealts``. This
dictionary is automatically tested for potential naming conflicts,
i.e., different properties that share the same alternative field
names.



How to use Ephem
----------------

As shown above (`How to use Ephem, Orbit, and Phys objects`_),
`~sbpy.data.Ephem` objects can be created on the fly. However,
`~sbpy.data.Ephem` can also be used to access ephemerides information
from remote services. For instance, the following few lines will query
ephemerides for asteroid Ceres on a given data and for the position of
Mauna Kea Observatory (IAU observatory code ``568``) from the `JPL Horizons service <https://ssd.jpl.nasa.gov/horizons.cgi>`_:

    >>> from sbpy.data import Ephem
    >>> from astropy.time import Time
    >>> epoch = Time('2018-08-03 14:20', scale='utc') # time in UT
    >>> eph = Ephem.from_horizons('Ceres',
    ...                           location='568',
    ...                           epochs=epoch)
    >>> print(eph) # doctest: +ELLIPSIS
    <sbpy.data.ephem.Ephem object at ...>
    >>> print(eph.table)
    targetname       datetime_str          datetime_jd    ...  PABLon PABLat
                                                d         ...   deg    deg  
    ---------- ------------------------ ----------------- ... ------- ------
       1 Ceres 2018-Aug-03 14:20:00.000 2458334.097222222 ... 171.275 9.3473
    >>> print(eph.column_names)
    <TableColumns names=('targetname','datetime_str','datetime_jd','H','G','solar_presence','flags','RA','DEC','RA_app','DEC_app','RA_rate','DEC_rate','AZ','EL','AZ_rate','EL_rate','sat_X','sat_Y','sat_PANG','siderealtime','airmass','magextinct','V','surfbright','illumination','illum_defect','sat_sep','sat_vis','ang_width','PDObsLon','PDObsLat','PDSunLon','PDSunLat','SubSol_ang','SubSol_dist','NPole_ang','NPole_dist','EclLon','EclLat','r','r_rate','delta','delta_rate','lighttime','vel_sun','vel_obs','elong','elongFlag','alpha','lunar_elong','lunar_illum','sat_alpha','sunTargetPA','velocityPA','OrbPlaneAng','constellation','TDB-UT','ObsEclLon','ObsEclLat','NPole_RA','NPole_DEC','GlxLon','GlxLat','solartime','earth_lighttime','RA_3sigma','DEC_3sigma','SMAA_3sigma','SMIA_3sigma','Theta_3sigma','Area_3sigma','RSS_3sigma','r_3sigma','r_rate_3sigma','SBand_3sigma','XBand_3sigma','DoppDelay_3sigma','true_anom','hour_angle','alpha_true','PABLon','PABLat')>

`~sbpy.data.Ephem.from_horizons` uses one or more target names, an
observer location in the form of an IAU observatory code, and a list
of discrete epochs or a range of epochs defined in a dictionary (see
`~sbpy.data.Ephem.from_horizons`) to query the JPL Horizons
service. Due to different requirements of the JPL Horizons service for
the epoch format, we recommend to use `~astropy.time.Time`
objects. The column names in the data table can be inquired using
`~sbpy.data.DataClass.column_names`.
    
`~sbpy.data.Ephem.from_horizons` is actually a wrapper around
`~astroquery.jplhorizons.HorizonsClass.ephemerides`. This function
conveniently combines the creation of a
`~astroquery.jplhorizons.HorizonsClass` query and the actual
ephemerides information retrieval into a single function. Additional
optional parameters provided to `~sbpy.data.Ephem.from_horizons` are
directly passed on to
`~astroquery.jplhorizons.HorizonsClass.ephemerides`, maintaining the
full flexibility of the latter function:

    >>> epoch1 = Time('2018-08-03 14:20', scale='utc')
    >>> epoch2 = Time('2018-08-04 07:30', scale='utc')
    >>> eph = Ephem.from_horizons('Ceres',
    ...                           location='568',
    ...                           epochs={'start': epoch1,
    ...                                   'stop': epoch2,
    ...                                   'step': '10m'},
    ...                           skip_daylight=True)
    >>> print(eph.table)
    targetname    datetime_str      datetime_jd    ... alpha_true  PABLon  PABLat
                                         d         ...    deg       deg     deg  
    ---------- ----------------- ----------------- ... ---------- -------- ------
       1 Ceres 2018-Aug-03 14:20 2458334.097222222 ...    12.9735  171.275 9.3473
       1 Ceres 2018-Aug-03 14:30 2458334.104166667 ...    12.9722 171.2774 9.3472
       1 Ceres 2018-Aug-03 14:40 2458334.111111111 ...     12.971 171.2798 9.3471
       1 Ceres 2018-Aug-03 14:50 2458334.118055556 ...    12.9698 171.2822  9.347
       1 Ceres 2018-Aug-03 15:00       2458334.125 ...    12.9685 171.2846 9.3469
       1 Ceres 2018-Aug-03 15:10 2458334.131944444 ...    12.9673 171.2869 9.3468
           ...               ...               ... ...        ...      ...    ...
       1 Ceres 2018-Aug-04 06:30 2458334.770833333 ...    12.8574 171.5052  9.337
       1 Ceres 2018-Aug-04 06:40 2458334.777777778 ...    12.8562 171.5076 9.3369
       1 Ceres 2018-Aug-04 06:50 2458334.784722222 ...     12.855 171.5099 9.3368
       1 Ceres 2018-Aug-04 07:00 2458334.791666667 ...    12.8538 171.5123 9.3367
       1 Ceres 2018-Aug-04 07:10 2458334.798611111 ...    12.8526 171.5147 9.3366
       1 Ceres 2018-Aug-04 07:20 2458334.805555556 ...    12.8513 171.5171 9.3365
       1 Ceres 2018-Aug-04 07:30      2458334.8125 ...    12.8501 171.5195 9.3364
       Length = 26 rows

Note that ``skip_daylight`` is an optional parameter of
`~astroquery.jplhorizons.HorizonsClass.ephemerides` and it can be used
here as well. An additional feature of
`~sbpy.data.Ephem.from_horizons` is that you can automatically
concatenate queries for a number of objects:

    >>> eph = Ephem.from_horizons(['Ceres', 'Pallas', 12893, '1983 SA'],
    ...                           location='568',
    ...                           epochs=epoch)
    >>> print(eph.table)
            targetname               datetime_str       ...  PABLon   PABLat 
                                                        ...   deg      deg   
    -------------------------- ------------------------ ... -------- --------
                       1 Ceres 2018-Aug-03 14:20:00.000 ...  171.275   9.3473
                      2 Pallas 2018-Aug-03 14:20:00.000 ... 132.9518 -20.1396
     12893 Mommert (1998 QS55) 2018-Aug-03 14:20:00.000 ... 100.9772  -2.0567
    3552 Don Quixote (1983 SA) 2018-Aug-03 14:20:00.000 ...   29.298  13.3365

Please be aware that these queries are not simultaneous. The more
targets you query, the longer the query will take. Furthermore, keep
in mind that asteroids and comets have slightly different table
layouts (e.g., different magnitude systems: ``T-mag`` and ``N-mag``
instead of ``V-mag``), which will complicate the interpretation of the
data. It might be safest to query asteroids and comets separately.

Ephemerides can also be derived from `~Orbit` objects using `sbpy`'s
interface to `pyoorb
<https://github.com/oorb/oorb/tree/master/python>`_ with the function
`~sbpy.data.Ephem.from_oorb`. The following example computes
ephemerides for the next ten days in steps of 1 hr for Ceres as seen
from the Discovery Channel Telescope:

     >>> import numpy as np
     >>> from sbpy.data import Orbit, Ephem
     >>> from astropy.time import Time
     >>> epochs = Time.now().jd + np.arange(0, 10, 1/24)
     >>> ceres = Orbit.from_horizons('1')
     >>> eph = Ephem.from_oo(ceres, 'G37', epochs)  # doctest: +SKIP
     >>> print(eph.table)  # doctest: +SKIP
     targetname      MJD [1]       ...        obsy [1]              obsz [1]       
                        d          ...           AU                    AU          
     ---------- ------------------ ... --------------------- ----------------------
        1 Ceres 58374.720415079966 ...   -0.1640418731222332 1.3660753531152814e-05
        1 Ceres  58374.76208174648 ...  -0.16334416599555382 1.6732994041007698e-05
        1 Ceres 58374.803748413455 ...  -0.16264729902661218 2.0200328928084155e-05
        1 Ceres 58374.845415079966 ...  -0.16195072092478624 2.3823231905778508e-05
        1 Ceres  58374.88708174648 ...  -0.16125385509757997  2.735153478080482e-05
        1 Ceres 58374.928748413455 ...  -0.16055613920683476 3.0541568772989025e-05
             ...                ... ...                   ...                    ...
        1 Ceres 58384.428748413455 ... 0.0016096754330388497  9.924120661052244e-06
        1 Ceres 58384.470415079966 ... 0.0023287044344341605   7.69766111133525e-06
        1 Ceres  58384.51208174648 ... 0.0030458232636104473  6.300640241761616e-06
        1 Ceres 58384.553748413455 ...  0.003760809893911351 5.8280310798125914e-06
        1 Ceres 58384.595415079966 ...  0.004473588211662766  6.311456253324348e-06
        1 Ceres  58384.63708174648 ...  0.005184233254950517  7.717021060406424e-06
        1 Ceres 58384.678748413455 ...  0.005892966025131429  9.947635868821306e-06
     Length = 240 rows

The properties computed by pyoorb and listed in the resulting table
are defined in the `pyoorb documentation
<https://github.com/oorb/oorb/tree/master/python>`_. Note that this function requires pyoorb to be installed, which is not a requirement for `sbpy`.
       
How to use Orbit
----------------

`~sbpy.data.Orbit.from_horizons` enables the query of Solar System
body osculating elements from the `JPL Horizons service
<https://ssd.jpl.nasa.gov/horizons.cgi>`_:

    >>> from sbpy.data import Orbit
    >>> from astropy.time import Time
    >>> epoch = Time('2018-05-14', scale='utc')
    >>> elem = Orbit.from_horizons('Ceres', epochs=epoch)
    >>> print(elem)  # doctest: +ELLIPSIS
    <sbpy.data.orbit.Orbit object at ...>
    >>> print(elem.table)
    targetname datetime_jd ...         Q                 P        
                    d      ...         AU                d        
    ---------- ----------- ... ----------------- -----------------
       1 Ceres   2458252.5 ... 2.976065555960228 1681.218128428134
    >>> print(elem.column_names)
    <TableColumns names=('targetname','datetime_jd','datetime_str','H','G','e','q','incl','Omega','w','Tp_jd','n','M','nu','a','Q','P')>

If ``epochs`` is not set, the osculating elements for the current
epoch (current time) are queried. Similar to
`~sbpy.data.Ephem.from_horizons`, this function is a wrapper for
`~astroquery.jplhorizons.HorizonsClass.elements` and passes optional
parameter on to that function. Furthermore, it is possible to query
orbital elements for a number of targets:

    >>> epoch = Time('2018-08-03 14:20', scale='utc')
    >>> elem = Orbit.from_horizons(['3749', '2009 BR60'],
    ...                            epochs=epoch,
    ...                            refplane='earth')
    >>> print(elem.table) # doctest:+ELLIPSIS
          targetname         datetime_jd    ...         Q                 P        
                                  d         ...         AU                d        
    --------------------- ----------------- ... ----------------- -----------------
    3749 Balam (1982 BG1) 2458334.097222222 ... 2.481...          1221.86...
       312497 (2009 BR60) 2458334.097222222 ... 2.481...          1221.77...

       
An existing `~Orbit` instance can be transformed to a different
orbital element definition system (e.g., Keplerian, cometary,
cartesian) using `~sbpy.data.Orbit.oo_transform` or it can be
propagated into the future or past using
`~sbpy.data.Orbit.oo_propagate`. Both functions are implemented in `sbpy` to provide an interface to `pyoorb
<https://github.com/oorb/oorb/tree/master/python>`_, a Python module using `OpenOrb <https://github.com/oorb/oorb>`_.

In order to transform some current orbits to a state vector in cartesian coordinates, one could use the following code:

    >>> elem = Orbit.from_horizons(['Ceres', 'Pallas', 'Vesta'])
    >>> statevec = elem.oo_transform('CART')  # doctest: +SKIP
    >>> print(statevec.table)  # doctest: +SKIP
       id             x                   y           ... epoch_scale  H    G  
                      AU                  AU          ...             mag      
    -------- ------------------- -------------------- ... ----------- ---- ----
     1 Ceres -2.5244444864469164 -0.35655744512932896 ...         UTC 3.34 0.12
    2 Pallas -1.7247978992440247   1.1268442378878316 ...         UTC 4.13 0.11
     4 Vesta  0.9453230885641772  -1.9811261840060208 ...         UTC  3.2 0.32
    
Orbits can currently be transformed to the following definitions: cartesian (``'CART'``), Keplerian (``'KEP'``), and cometary (``'COM'``).

Orbit propagation requires the epoch to which the orbit should be propagated to either as `~astropy.time.Time` object, or as float in terms of Julian date. The following example propagates the current orbit of Ceres back to year 2000:

    >>> elem = Orbit.from_horizons('Ceres')
    >>> epoch = Time('2000-01-01', format='iso')
    >>> newelem = elem.oo_propagate(epoch)  # doctest: +SKIP
    >>> print(newelem.table)  # doctest: +SKIP
       id           a                  e          ... epoch_scale  H    G  
                    AU                            ...             mag      
    ------- ----------------- ------------------- ... ----------- ---- ----
    1 Ceres 2.766494213803344 0.07837503821706865 ...         UTC 3.34 0.12

Note that both functions require pyoorb to be installed, which is not a requirement for `sbpy`.
    
How to use Phys
---------------
tbd


How to use Names
----------------

`~sbpy.data.Names` is different from the other classes in `~sbpy.data`
in that it does not use `~sbpy.data.DataClass` as a base class. Instead,
`~sbpy.data.Names` does not contain any data, it merely serves as an
umbrella for functions to identify asteroid and comet names, numbers,
and designations.

In order to distinguish if a string designates a comet or an asteroid,
you can use the following code:

    >>> from sbpy.data import Names
    >>> print(Names.asteroid_or_comet('(1) Ceres'))
    asteroid
    >>> print(Names.asteroid_or_comet('2P/Encke'))
    comet

The module basically uses regular expressions to match the input
strings and find patterns that agree with asteroid and comet names,
numbers, and designations. There are separate tasks to identify
asteroid and comet identifiers:

    >>> print(Names.parse_asteroid('(228195) 6675 P-L')) # doctest: +SKIP
    {'number': 228195, 'desig': '6675 P-L'}
    >>> print(Names.parse_asteroid('C/2001 A2-A (LINEAR)')) # doctest: +SKIP
    ... sbpy.data.names.TargetNameParseError: C/2001 A2-A (LINEAR) does not appear to be an asteroid identifier
    >>> print(Names.parse_comet('12893')) # doctest: +SKIP
    ... sbpy.data.names.TargetNameParseError: 12893 does not appear to be a comet name
    >>> print(Names.parse_comet('73P-C/Schwassmann Wachmann 3 C	')) # doctest: +SKIP
    {'type': 'P', 'number': 73, 'fragment': 'C', 'name': 'Schwassmann Wachmann 3 C'}
    
In order to be able to distinguish between asteroid and comet
identifiers, `sbpy` follows the MPC guideline in that it requires
comet identifiers to include the comet type in either in combination
with a number (e.g., ``'259P'``), a name (e.g., ``'P/Halley'``), or
both (e.g., ``'2P/Encke'``). For instance, the identifier ``'Halley'``
would be identified as an asteroid, as it lacks a comet type
identifier. Hence, some caution is advised when using these routines -
identification might not be unambiguous.

    


Reference/API
-------------
.. automodapi:: sbpy.data
    :no-heading:
		


 
