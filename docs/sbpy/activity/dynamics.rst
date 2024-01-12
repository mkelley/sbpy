Dust dynamical models
=====================

`sbpy` provides classes that may be used to solve the equations of motion of dust.  This sub-module supports the generation of syndynes (see :ref:`syndynes`).

.. _state-objects:

State objects
-------------

`sbpy` uses `~sbpy.activity.dust.dynamics.State` objects to encapsulate the position and velocity of an object at a given time.  Create a `~sbpy.activity.dust.dynamics.State` for a comet at :math:`x=2` au, moving along the y-axis at a speed of 30 km/s:

.. doctest::

    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from sbpy.activity.dust import State
    >>> 
    >>> r = [2, 0, 0] * u.au
    >>> v = [0, 30, 0] * u.km / u.s
    >>> t = Time("2023-12-08")
    >>> comet = State(r, v, t)

Time can be specified with `~astropy.time.Time` objects, as above, or as an arbitrary `~astropy.units.Quantity` with units of time:

   >>> comet = State(r, v, 0 * u.s)

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

Mathematical Operations
^^^^^^^^^^^^^^^^^^^^^^^

A few simple mathematical operations are possible:

.. doctest::

    >>> observer = State([1, 1, 0] * u.au, [15, -15, 0] * u.km / u.s, comet.t)
    >>> obs_comet = comet - observer
    >>> obs_comet.r
    <Quantity [ 1.49597871e+08, -1.49597871e+08,  0.00000000e+00] km>
    >>> obs_comet.v
    <Quantity [-15.,  45.,   0.] km / s>
    >>> abs(obs_comet)
    (<Quantity 2.11563338e+08 km>, <Quantity 47.4341649 km / s>)

To/From `SkyCoord`
^^^^^^^^^^^^^^^^^^

With coordinate frames specified and time provided with `~astropy.time.Time` objects, states may be converted to `~astropy.coordinates.SkyCoord` objects:

.. doctest::

    >>> comet.frame = "icrs"
    >>> coords = comet.to_skycoord()
    >>> coords
    <SkyCoord (ICRS): (x, y, z) in km
        (2.99195741e+08, 0., 0.)
     (v_x, v_y, v_z) in km / s
        (0., 30., 0.)>
    >>>
    >>> coords.representation_type = "spherical"
    >>> coords
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, km)
        (0., 0., 2.99195741e+08)
     (pm_ra, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (6.52671948e+08, 0., 0.)>

`States` may be generated from `SkyCoord` objects if they contain position, velocity, and time coordinates:

    >>> from astropy.coordinates import SkyCoord
    >>> coords = SkyCoord("1:23:45h -6:07:08d",
    ...                   distance=1 * u.au,
    ...                   pm_ra_cosdec=100 * u.arcsec / u.hr,
    ...                   pm_dec=0 * u.arcsec / u.hr,
    ...                   obstime=Time("2023-12-11"),
    ...                   frame="icrs")
    >>> state = State.from_skycoord(coords)

Observing `States`
^^^^^^^^^^^^^^^^^^

the relative coordinate between two states can be computed with :meth:`~sbpy.activity.dust.dynamics.State.observe`:

.. doctest::

    >>> observer.frame = "icrs"
    >>> observer.observe(comet)
    <SkyCoord (ICRS): (ra, dec, distance) in (deg, deg, km)
        (315., 0., 2.11563338e+08)
     (pm_ra, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (6.52671948e+08, 0., -42.42640687)>

Dynamical models
----------------

`sbpy`'s built-in models solve the equations of motion for dust grains given two-body dynamics:

* `sbpy.activity.dust.dynamics.FreeExpansion` solves for linear motion.
* `sbpy.activity.dust.dynamics.SolarGravity` solves for orbits around the Sun.
* `sbpy.activity.dust.dynamics.SolarGravityAndRadiationPressure` solves for orbits around the Sun, including the effects of solar radiation pressure.

The following example shows how the models may be used by plotting the trajectories of test particles:

.. plot::
    :include-source:

    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    from sbpy.activity.dust import (State,
                                    FreeExpansion,
                                    SolarGravity,
                                    SolarGravityAndRadiationPressure)
    
    rh = 1 * u.au
    speed = np.sqrt(SolarGravity().GM / rh).to("km / s")
    period = 2 * np.pi * rh / speed

    initial = State(rh * np.array([1, 0, 0]), speed * np.array([0, 1, 0]), 0 * u.s)

    def plot(ax, solver, t_f, *solver_args, **plot_kwargs):
        final = []
        for t in np.linspace(0, t_f):
            final.append(solver.solve(initial, t, *solver_args))
        final = State.from_states(final)
        ax.plot(final.x.to("au"), final.y.to("au"), **plot_kwargs)

    fig, ax = plt.subplots()
    plot(ax, FreeExpansion(), period, ls="-", label="FreeExpansion")
    plot(ax, SolarGravity(), period, ls="--", label="SolarGravity")
    plot(ax, SolarGravityAndRadiationPressure(), 1.3 * period,
         0.1, ls="-.", label="SolarGravityAndRadiationPressure, beta=0.1")
    plt.setp(ax, aspect="equal",
             xlim=[-1.5, 1.3], ylim=[-1.2, 1.2],
             xlabel="$X$ (au)", ylabel="$Y$ (au)")
    plt.legend()
    plt.tight_layout()


Users may provide their own models in order to, e.g., improve code performance, or add planetary perturbations.  Subclass `~sbpy.activity.dust.dynamics.DynamicalModel` and override the ``df_dt`` and ``df_drv`` methods.