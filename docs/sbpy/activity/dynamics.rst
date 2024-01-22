Dust dynamics (`sbpy.activity.dust.dynamics`)
=============================================

`sbpy` has the capability to integrate test particle orbits, primarily intended to support the generation of :ref:`comet syndynes <sbpy/activity/dust:dust syndynes and synchrones>`.


Dynamical state 
---------------

`~sbpy.activity.dust.dynamics.State` objects encapsulate the position and velocity of an object at a given time.  Create a `~sbpy.activity.dust.dynamics.State` for a comet at :math:`x=2` au, moving along the y-axis at a speed of 30 km/s:


.. doctest::

    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from sbpy.activity.dust import State
    >>> 
    >>> r = [2, 0, 0] * u.au
    >>> v = [0, 30, 0] * u.km / u.s
    >>> t = Time("2023-12-08")
    >>> comet = State(r, v, t)

`~sbpy.activity.dust.dynamics.State` objects may also represent an array of objects:

.. doctest::

    >>> r = ([2, 0, 0], [0, 2, 0]) * u.au
    >>> v = ([0, 30, 0], [30, 0, 0]) * u.km / u.s
    >>> t = Time(["2023-12-08", "2023-12-09"])
    >>> comets = State(r, v, t)
    >>> len(comets)
    2

The `r`, `v`, and `t` attributes hold the position, velocity, and time for the object(s).  The first index iterates over the object, the second iterates over the x-, y-, and z-axes.

.. doctest::

    >>> comets.r.shape
    (2, 3)
    >>> comets.r[0]
    <Quantity [2.99195741e+08, 0.00000000e+00, 0.00000000e+00] km>

Or, index the `State` object itself:

.. doctest::

    >>> comets[0].r  # equivalent to comets.r[0]
    <Quantity [2.99195741e+08, 0.00000000e+00, 0.00000000e+00] km>


Convert to/from ``Ephem`` and ``SkyCoord``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

State objects may be initialized from ephemeris objects (`~sbpy.data.Ephem`), provided they contain time, and 3D position and velocity:

.. doctest-requires:: astroquery

    >>> from sbpy.data import Ephem
    >>> eph = Ephem.from_horizons("9P",
    ...                           epochs=Time("2005-07-04"),
    ...                           id_type="designation",
    ...                           closest_apparition=True)  # doctest: +REMOTE_DATA
    >>> tempel1 = State.from_ephem(eph)                     # doctest: +REMOTE_DATA

And ``State`` may be converted back to an ``Ephem`` object:

.. doctest-requires:: astroquery

    >>> eph = tempel1.to_ephem()  # doctest: +REMOTE_DATA

`astropy`'s `~astropy.coordinates.SkyCoord` objects may also be used, assuming the time and 3D vectors are fully defined:

    >>> from astropy.coordinates import SkyCoord
    >>> coords = SkyCoord(ra="01:02:03h",
    ...                   dec="+04:05:06d",
    ...                   pm_ra_cosdec=100 * u.arcsec / u.hr,
    ...                   pm_dec=-10 * u.arcsec / u.hr,
    ...                   distance=1 * u.au,
    ...                   radial_velocity=15 * u.km / u.s,
    ...                   obstime=Time("2024-01-19"))
    >>> state = State.from_skycoord(coords)

And back to a ``SkyCoord`` object:

.. doctest::

    >>> state.to_skycoord()
    <SkyCoord (ICRS): (x, y, z) in km
        (1.43782125e+08, 39908095.19296389, 10656800.66639306)
     (v_x, v_y, v_z) in km / s
        (9.16701924, 23.45244422, -0.94097856)>


Reference frames
^^^^^^^^^^^^^^^^

Coordinate reference frames can be specified with the ``frame`` keyword argument.  Most ``astropy`` reference frames are supported (see `astropy's Built-in Frame Classes <https://docs.astropy.org/en/stable/coordinates/index.html#module-astropy.coordinates.builtin_frames>`):

.. note::
    When working with heliocentric ecliptic coordinates from JPL Horizons or NAIF SPICE, you may want to use the `~astropy.coordinates.HeliocentricEclipticIAU76` reference frame.

.. doctest::

    >>> r = [2, 0, 0] * u.au
    >>> v = [0, 30, 0] * u.km / u.s
    >>> t = Time("2023-12-08")
    >>> state = State(r, v, t, frame="heliocentriceclipticiau76")
    >>> state
    <State (<HeliocentricEclipticIAU76 Frame (obstime=J2000.000)>):
     r
        [2.99195741e+08 0.00000000e+00 0.00000000e+00] km
     v
        [ 0. 30.  0.] km / s
     t
        755265669.183221>

Use :func:`~sbpy.activity.dust.dynamics.State.transform_to` to transform between reference frames:

.. doctest::

    >>> new_state = state.transform_to("icrs")
    >>> new_state
    <State (<ICRS Frame>):
     r
        [ 2.97986677e+08 -3.87811613e+05 -1.33685640e+05] km
     v
        [8.11760173e-03  2.75129298e+01  1.19282350e+01] km / s
     t
        755265669.183221>


State lengths, subtraction, and observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A few mathematical operations are possible.  Get the magnitude of the heliocentric distance and velocity with `abs`:

.. doctest::

    >>> print(abs(comet))
    (<Quantity 2.99195741e+08 km>, <Quantity 30. km / s>)

Get the Earth-comet state vector by subtraction:

.. doctest::

    >>> earth = State([0, 1, 0] * u.km, [30, 0, 0] * u.km / u.s, comet.t)
    >>> print(comet - earth)
    <State (None):
     r
        [ 2.99195741e+08 -1.00000000e+00  0.00000000e+00] km
     v
        [-30.  30.   0.] km / s
     t
        755265669.183221>

Or, get the sky coordinates of the comet as seen from the Earth.  Because this creates a `~astropy.coordinates.SkyCoord` object, a reference frame is required:

.. doctest::

    >>> comet.frame = "icrs"
    >>> earth.frame = "icrs"
    >>> earth.observe(comet)


Dynamical integrators
---------------------

A state object may be propagated to a new time using a dynamical integrator.  Three integrators are defined.  Use `~sbpy.activity.dust.dynamics.FreeExpansion` for motion in free space, `~sbpy.activity.dust.dynamics.SolarGravity` for orbits around the Sun, and `~sbpy.activity.dust.dynamics.SolarGravityAndRadiationPressure`  for orbits around the Sun considering radiation pressure.

.. doctest-requires:: scipy

    >>> from sbpy.activity.dust import SolarGravity
    >>> state = State([1, 0, 0] * u.au, [0, 30, 0] * u.km / u.s, 0 * u.s)
    >>> integrator = SolarGravity()
    >>> t_final = 1 * u.year
    >>> integrator.solve(state, t_final)
    <State (None):
     r
        [ 1.4814690e+08 -2.0935988e+07  0.0000000e+00] km
     v
        [ 4.13782154 29.70907118  0.        ] km / s
     t
        31557600.0 s>


.. automodapi:: sbpy.activity.dust.dynamics