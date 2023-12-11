Dust syndynes and synchrones
============================

Syndynes are lines in space connecting particles that are experiencing the same forces.  A syndyne is parameterized by :math:`\beta`, the ratio of the force from solar radiation to the force from solar gravity, :math:`F_r / F_g`, and age (or time of release).  Thus, all particles in a syndyne have a constant :math:`\beta` but variable age.  Similarly, synchrones are lines of constant particle age, but variable :math:`\beta`.

.. _syndynes:

Syndynes
--------

Syndynes are generated with the `~sbpy.activity.dust.syndynes.Syndynes` class.  The class requires a `~sbpy.activity.dust.dynamics.State` object (see :ref:`state-objects`), representing the source of the syndynes, a list of :math:`\beta` values and particle ages from which to generate the syndynes.

First, define the source of the syndynes, a comet at 2 au from the Sun:

.. doctest::

   >>> from astropy.time import Time
   >>> import astropy.units as u
   >>> from sbpy.activity.dust import State
   >>> 
   >>> r = [2, 0, 0] * u.au
   >>> v = [0, 30, 0] * u.km / u.s
   >>> t = Time("2023-12-08")
   >>> comet = State(r, v, t)

Next, initialize the syndyne object:

.. doctest::

   >>> import numpy as np
   >>> from sbpy.activity.dust import Syndynes
   >>> 
   >>> betas = [1, 0.1, 0.01, 0]
   >>> ages = np.linspace(0, 100, 25) * u.day
   >>> syn = Syndynes(comet, betas, ages)
   >>> syn
   <Syndynes:
   betas
      [1.   0.1  0.01 0.  ]
   ages
      [      0.  360000.  720000. 1080000. 1440000. 1800000. 2160000. 2520000.
   2880000. 3240000. 3600000. 3960000. 4320000. 4680000. 5040000. 5400000.
   5760000. 6120000. 6480000. 6840000. 7200000. 7560000. 7920000. 8280000.
   8640000.] s>

To compute the syndynes, use the :meth:`~sbpy.activity.dust.syndynes.Syndynes.solve` method.  The computed particle positions are saved in the :attr:`~sbpy.activity.dust.syndynes.Syndynes.particles` attribute.  For our example, the 4 :math:`\beta`-values and the 50 ages produce 150 particles:

.. doctest::

   >>> syn.solve()
   >>> print(len(syn.particles))
   100

Inspect the results using :meth:`~sbpy.activity.dust.syndynes.Syndynes.syndynes`, which returns an iterator containing each syndyne's :math:`\beta`-value and particle states.  For example, we can compute the maximum linear distance from the comet to the syndyne particles:

.. doctest::

   >>> for beta, states in syn.syndynes():
   ...     r, v = abs(states - comet)
   ...     print("{:.3f}".format(r.max().to("au")))
   0.309 AU
   0.032 AU
   0.003 AU
   0.000 AU

Individual syndynes may be produced with the :meth:`~sbpy.activity.dust.syndynes.Syndynes.get_syndyne` method and a syndyne index.  The index for the syndyne matches the index of the ``betas`` array.  To get the :math:`\beta=0.1` syndyne from our example:

.. doctest::

   >>> print(syn.betas)
   [1.   0.1  0.01 0.  ]
   >>> beta, states = syn.get_syndyne(1)
   >>> print(beta)
   0.1

Synchrones
----------

Synchrones are also simulated with the `~sbpy.activity.dust.syndynes.Syndynes` class, but instead generated with the :meth:`~sbpy.activity.dust.syndynes.Syndynes.get_synchrone` and :meth:`~sbpy.activity.dust.syndynes.Syndynes.synchrones` methods.

.. doctest::

   >>> age, states = syn.get_synchrone(24)
   >>> r, v = abs(states)
   >>> print(age, "{:.3g}".format(r.max().to("au")))
   8640000.0 s 2.27 AU


Projecting onto the sky
-----------------------

Syndynes and synchrones may be projected onto the sky as seen by a telescope.  This requires an observer and sky coordinate frames.  `sbpy` uses `astropy`'s reference frames, which may be specified as a string or an instance of a reference frame object.  For precision work, the states provided to the `~sbpy.activity.dust.syndynes.Syndynes` object should be in a heliocentric coordinate frame.  Here, we use a J2000 heliocentric ecliptic coordinate frame that <JPL Horizons `https://ssd.jpl.nasa.gov/horizons/manual.html#frames`>_ and the <NAIF SPICE toolkit `https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html#Frames%20Supported%20in%20SPICE`>_: `"heliocentriceclipticiau76"`:

.. doctest::

   >>> comet.frame = "heliocentriceclipticiau76"
   >>> observer = State(
   ...     r=[0, 2, 2] * u.au,
   ...     v=[0, 0, 0] * u.km / u.s,
   ...     t=comet.t,
   ...     frame="heliocentriceclipticiau76"
   ... )
   >>> syn = Syndynes(comet, betas, ages, observer=observer)
   >>> syn.solve()

With the observer and coordinate frames defined, the syndyne and synchrone methods will return `astropy.coordinates.SkyCoord` objects that represent the sky positions of the test particles.  Here, we request the coordinate object is returned in an ICRS-based reference frame and print a sample of the coordinates:

.. doctest::

   >>> beta, states, coords = syn.get_syndyne(0, frame="icrs")
   >>> print("\n".join(coords[::5].to_string("hmsdms", precision=0)))
   22h10m09s -49d24m30s
   22h10m44s -49d13m51s
   22h11m44s -48d43m46s
   22h12m11s -47d59m54s
   22h11m27s -47d09m25s

Source object orbit
-------------------

Calculating the positions of the projected orbit of the source object may be helpful for interpreting an observation or a set of syndynes.  They are calculated with the :meth:`~sbpy.activity.dust.synydnes.Syndynes.get_orbit` method:

.. doctest::

   >>> dt = np.linspace(-2, 2) * u.d
   >>> orbit, coords = syn.get_orbit(dt, frame="icrs")

Other dynamical models
----------------------

In this example, we compute the syndynes of a comet orbiting β Pic (1.8 solar masses) by sub-classing `~sbpy.activity.dust.dynamics.SolarGravityAndRadiationPressure` and updating :math:`GM`, the mass of the star times the gravitational constant:

.. doctest::

   >>> import astropy.constants as const
   >>> from sbpy.activity.dust import SolarGravityAndRadiationPressure
   >>>
   >>> class BetaPicGravityAndRadiationPressure(SolarGravityAndRadiationPressure):
   ...     _GM = 1.8 * SolarGravityAndRadiationPressure._GM
   >>>
   >>> solver = BetaPicGravityAndRadiationPressure()
   >>> betapic_syn = Syndynes(comet, [1, 0], [0, 100] * u.d, solver=solver)


Plotting syndynes and synchrones
--------------------------------

Generally, we are interested in plotting syndynes and synchrones on an image of a comet.  The accuracy of the coordinates object depends on the the comet and observer states, but also on whether or not light travel time is accounted for.  The `sbpy` testing suite shows that arcsecond-level accuracy is possible, but this is generally not accurate enough for direct comparison to typical images of comets.  Instead, it helps to compute the positions of the syndynes and synchrone coordinate objects relative to the comet, and plot the results.

.. doctest::

   >>> from itertools import islice
   >>> import matplotlib.pyplot as plt
   >>> 
   >>> coords0 = observer.observe(comet, frame="icrs")
   >>> def plot(ax, coords, **kwargs):
   ...     dRA = coords.ra - coords0.ra
   ...     dDec = coords.dec - coords0.dec
   ...     ax.plot(dRA.arcsec, dDec.arcsec, **kwargs)
   >>> 
   >>> fig, ax = plt.subplots()
   >>> 
   >>> for beta, states, coords in syn.syndynes("icrs"):
   ...     # don't draw the beta = 0 syndyne
   ...     if beta == 0:
   ...         continue
   ...     plot(ax, coords, label=f"$\\beta={beta:.2g}$")
   >>> 
   >>> # use islice to plot every 5th synchrone
   >>> for age, states, coords in islice(syn.synchrones("icrs"), 4, None, 5):
   ...     plot(ax, coords, ls="--", label=f"$\\Delta t={age.to(u.d):.2g}$")
   >>> 
   >>> # and plot the orbit
   >>> dt = np.linspace(-2, 2) * u.d
   >>> states, coords = syn.get_orbit(dt, frame="icrs")
   >>> plot(ax, coords, color="k", ls=":", label="Orbit")
   >>>
   >>> ax.invert_xaxis()
   >>> plt.setp(ax,
   ...          xlim=[100, -10],
   ...          ylim=[-10, 100],
   ...          xlabel="$\\Delta$RA (arcsec)",
   ...          ylabel="$\\Delta$Dec (arcsec)",
   ... ) # doctest: +SKIP
   >>> plt.legend() # doctest: +SKIP
   >>> plt.tight_layout()

.. plot::

   from itertools import islice

   import numpy as np
   import matplotlib.pyplot as plt

   import astropy.units as u
   from astropy.time import Time
   from sbpy.activity.dust import State, Syndynes

   r = [2, 0, 0] * u.au
   v = [0, 30, 0] * u.km / u.s
   t = Time("2023-12-08")
   frame = "heliocentriceclipticiau76"
   comet = State(r, v, t, frame=frame)

   betas = [1, 0.1, 0.01, 0]
   ages = np.linspace(0, 100, 25) * u.day
   observer = State(
       r=[0, 2, 2] * u.au,
       v=[0, 0, 0] * u.km / u.s,
       t=comet.t,
       frame=frame,
   )
   syn = Syndynes(comet, betas, ages, observer=observer)
   syn.solve()

   coords0 = observer.observe(comet, frame="icrs")
   def plot(ax, coords, **kwargs):
       dRA = coords.ra - coords0.ra
       dDec = coords.dec - coords0.dec
       ax.plot(dRA.arcsec, dDec.arcsec, **kwargs)
   
   fig, ax = plt.subplots()
   
   for beta, states, coords in syn.syndynes("icrs"):
       # don't draw the beta = 0 syndyne
       if beta == 0:
           continue
       plot(ax, coords, label=f"$\\beta={beta:.2g}$")
   
   # use islice to plot every 5th synchrone
   for age, states, coords in islice(syn.synchrones("icrs"), 4, None, 5):
       plot(ax, coords, ls="--", label=f"$\Delta t={age.to(u.d):.2g}$")
   
   # and plot the orbit
   dt = np.linspace(-2, 2) * u.d
   states, coords = syn.get_orbit(dt, frame="icrs")
   plot(ax, coords, color="k", ls=":", label="Orbit")

   ax.invert_xaxis()
   plt.setp(ax,
            xlim=[100, -10],
            ylim=[-10, 100],
            xlabel="$\Delta$RA (arcsec)",
            ylabel="$\Delta$Dec (arcsec)",
   )
   plt.legend()
   plt.tight_layout()


Reference/API
-------------
.. automodapi:: sbpy.activity.dust.syndynes
   :no-heading:
