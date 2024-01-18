Dust comae and tails (`sbpy.activity.dust`)
===========================================

Cometary dust is the refractory material released by comets.  This sub-module provides simple photometric models of cometary dust comae, and dynamical models for cometary dust tails.

.. contents::
   :local:

*Afρ* and *εfρ* models
----------------------

`sbpy` has two classes to support observations and models of a coma continuum: `~sbpy.activity.dust.core.Afrho` and `~sbpy.activity.dust.core.Efrho`.

The *Afρ* parameter of A'Hearn et al (1984) is based on observations of idealized cometary dust comae.  It is proportional to the observed flux density within a circular aperture.  The quantity is the product of dust albedo, dust filling factor, and the radius of the aperture at the distance of the comet.  It carries the units of *ρ* (length), and under certain assumptions is proportional to the dust production rate of the comet:

.. math::

   Afρ = \frac{4 Δ^2 r_h^2}{ρ}\frac{F_c}{F_⊙}

where *Δ* and *ρ* have the same (linear) units, but :math:`r_h` is in units of au.  :math:`F_c` * is the flux density of the comet in the aperture, and :math:`F_⊙` is that of the Sun at 1 au in the same units.  See A'Hearn et al. (1984) and Fink & Rubin (2012) for more details.

The *εfρ* parameter is the thermal emission counterpart to *Afρ*, replacing albedo with IR emissivity, *ε*, and the solar spectrum with the Planck function, *B*:

.. math::

   εfρ = \frac{F_c Δ^2}{π ρ B(T_c)}

where :math:`T_c` is the spectral temperature of the continuum (Kelley et al. 2013).

*Afρ* and *εfρ* are quantities
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`~sbpy.activity.dust.core.Afrho` and `~sbpy.activity.dust.core.Efrho` are subclasses of `astropy`'s `~astropy.units.Quantity` and carry units of length.

   >>> import numpy as np
   >>> import astropy.units as u
   >>> from sbpy.activity import Afrho, Efrho
   >>>
   >>> afrho = Afrho(100 * u.cm)
   >>> print(afrho)    # doctest: +FLOAT_CMP
   100.0 cm
   >>> efrho = Efrho(afrho * 3.5)
   >>> print(efrho)    # doctest: +FLOAT_CMP
   350.0 cm

They may be converted to other units of length just like any `~astropy.units.Quantity`:

   >>> afrho.to('m')    # doctest: +FLOAT_CMP
   <Afrho 1. m>

.. _afrho-to-from-flux-density:

Convert to/from flux density
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The quantities may be initialized from flux densities.  Here, we reproduce one of the calculations from the original A'Hearn et al. (1984) work:

* :math:`r_h` = 4.785 au
* *Δ* = 3.822 au
* phase angle = 3.3°
* aperture radius = 9.8" (27200 km)
* :math:`\log{F_λ}` = -13.99 erg/(cm\ :sup:`2` s Å) at *λ* = 5240 Å

The solar flux density at 1 au is also needed.  We use 1868 W/(m2 μm).

   >>> from sbpy.data import Ephem
   >>> from sbpy.calib import solar_fluxd
   >>>
   >>> solar_fluxd.set({
   ...     'λ5240': 1868 * u.W / u.m**2 / u.um,
   ...     'λ5240(lambda pivot)': 5240 * u.AA
   ... })              # doctest: +IGNORE_OUTPUT
   >>>
   >>> flam = 10**-13.99 * u.Unit('erg/(s cm2 AA)')
   >>> aper = 27200 * u.km
   >>>
   >>> eph = Ephem.from_dict({'rh': 4.785 * u.au, 'delta': 3.822 * u.au})
   >>>
   >>> afrho = Afrho.from_fluxd('λ5240', flam, aper, eph)
   >>> print(afrho)    # doctest: +FLOAT_CMP
   6029.90248952895 cm

Which is within a few percent of 6160 cm computed by A'Hearn et al.. The difference is likely due to the assumed solar flux density in the bandpass.

The `~sbpy.activity.dust.core.Afrho` class may be converted to a flux density, and the original value is recovered.

   >>> f = afrho.to_fluxd('λ5240', aper, eph).to('erg/(s cm2 AA)')
   >>> print(np.log10(f.value))    # doctest: +FLOAT_CMP
   -13.99

`~sbpy.activity.dust.core.Afrho` works seamlessly with `sbpy`'s spectral calibration framework (:ref:`sbpy-calib`) when the `astropy` affiliated package `synphot` is installed.  The solar flux density (via `~sbpy.calib.solar_fluxd`) is not required, but instead the spectral wavelengths or the system transmission of the instrument and filter:

.. doctest-requires:: synphot; astropy>=5.3

   >>> wave = [0.4, 0.5, 0.6] * u.um
   >>> print(afrho.to_fluxd(wave, aper, eph))    # doctest: +FLOAT_CMP
   [7.76770018e-14 1.05542873e-13 9.57978939e-14] W / (um m2)

.. doctest-requires:: synphot

   >>> from synphot import SpectralElement, Box1D
   >>> # bandpass width is a guess
   >>> bp = SpectralElement(Box1D, x_0=5240 * u.AA, width=50 * u.AA)
   >>> print(Afrho.from_fluxd(bp, flam, aper, eph))    # doctest: +FLOAT_CMP
   5994.110239075767 cm

Thermal emission with *εfρ*
^^^^^^^^^^^^^^^^^^^^^^^^^^^
   
The `~sbpy.activity.dust.core.Efrho` class has the same functionality as the `~sbpy.activity.dust.core.Afrho` class.  The most important difference is that *εfρ* is calculated using a Planck function and temperature.  `sbpy` follows common practice and parameterizes the temperature as a constant scale factor of :math:`T_{BB} = 278\,r_h^{1/2}`\  K, the equilibrium temperature of a large blackbody sphere at a distance :math:`r_h` from the Sun.

Reproduce the *εfρ* of 246P/NEAT from Kelley et al. (2013).

.. doctest-requires:: synphot

   >>> wave = [15.8, 22.3] * u.um
   >>> fluxd = [25.75, 59.2] * u.mJy
   >>> aper = 11.1 * u.arcsec
   >>> eph = Ephem.from_dict({'rh': 4.28 * u.au, 'delta': 3.71 * u.au})
   >>> efrho = Efrho.from_fluxd(wave, fluxd, aper, eph)
   >>> for i in range(len(wave)):
   ...     print('{:5.1f} at {:.1f}'.format(efrho[i], wave[i]))    # doctest: +FLOAT_CMP
   406.2 cm at 15.8 um
   427.9 cm at 22.3 um

Compare to 397.0 cm and 424.6 cm listed in Kelley et al. (2013).


To/from magnitudes
^^^^^^^^^^^^^^^^^^

`~sbpy.activity.dust.core.Afrho` and `~sbpy.activity.dust.core.Efrho` also work with `astropy`'s magnitude units.  If the conversion between Vega-based magnitudes is required, then `sbpy`'s calibration framework (:ref:`sbpy-calib`) will be used.

Estimate the *Afρ* of comet C/2012 S1 (ISON) based on Pan-STARRS 1 photometry in the *r* band (Meech et al. 2013)

.. doctest-requires:: synphot

   >>> w = 0.617 * u.um
   >>> m = 16.02 * u.ABmag
   >>> aper = 5 * u.arcsec
   >>> eph = {'rh': 5.234 * u.au, 'delta': 4.275 * u.au, 'phase': 2.6 * u.deg}
   >>> afrho = Afrho.from_fluxd(w, m, aper, eph)
   >>> print(afrho)    # doctest: +FLOAT_CMP
   1948.496075629444 cm
   >>> m2 = afrho.to_fluxd(w, aper, eph, unit=u.ABmag)    # doctest: +FLOAT_CMP
   >>> print(m2)
   16.02 mag(AB)


Phase angles and functions
^^^^^^^^^^^^^^^^^^^^^^^^^^

Phase angle was not used in the previous section.  In the *Afρ* formalism, "albedo" includes the scattering phase function, and is more precisely written *A(θ)*, where *θ* is the phase angle.  The default behavior for `~sbpy.activity.dust.core.Afrho` is to compute *A(θ)fρ* as opposed to *A(0°)fρ*.  Returning to the A'Hearn et al. data, we scale *Afρ* to 0° from 3.3° phase using the :func:`~sbpy.activity.Afrho.to_phase` method:

.. doctest-requires:: scipy

   >>> afrho = Afrho(6029.9 * u.cm)
   >>> print(afrho.to_phase(0 * u.deg, 3.3 * u.deg))    # doctest: +FLOAT_CMP
   6886.825981017757 cm

The default phase function is the Halley-Marcus composite phase function (:func:`~sbpy.activity.phase_HalleyMarcus`).  Any function or callable object that accepts an angle as a `~astropy.units.Quantity` and returns a scalar value may be used:

.. doctest-requires:: scipy

   >>> Phi = lambda phase: 10**(-0.016 / u.deg * phase.to('deg'))
   >>> print(afrho.to_phase(0 * u.deg, 3.3 * u.deg, Phi=Phi))    # doctest: +FLOAT_CMP
   6809.419810008357 cm

To correct an observed flux density for the phase function, use the ``phasecor`` option of :func:`~sbpy.activity.Afrho.to_fluxd` and :func:`~sbpy.activity.Afrho.from_fluxd` methods:

.. doctest-requires:: scipy

   >>> flam = 10**-13.99 * u.Unit('erg/(s cm2 AA)')
   >>> aper = 27200 * u.km
   >>> eph = Ephem.from_dict({
   ...     'rh': 4.785 * u.au,
   ...     'delta': 3.822 * u.au,
   ...     'phase': 3.3 * u.deg
   ... })
   >>>
   >>> afrho = Afrho.from_fluxd('λ5240', flam, aper, eph, phasecor=True)
   >>> print(afrho)    # doctest: +FLOAT_CMP
   6886.828824340642 cm


Using apertures
^^^^^^^^^^^^^^^

Other apertures may be used, as long as they can be converted into an equivalent radius, assuming a coma with a *1/ρ* surface brightness distribution.  `~sbpy.activity` has a collection of useful geometries.

   >>> from sbpy.activity import CircularAperture, AnnularAperture, RectangularAperture, GaussianAperture
   >>> apertures = (
   ...   ( '10" radius circle', CircularAperture(10 * u.arcsec)),
   ...   (    '5"–10" annulus', AnnularAperture([5, 10] * u.arcsec)),
   ...   (       '2"x10" slit', RectangularAperture([2, 10] * u.arcsec)),
   ...   ('σ=5" Gaussian beam', GaussianAperture(5 * u.arcsec))
   ... )
   >>> for name, aper in apertures:
   ...     afrho = Afrho.from_fluxd('λ5240', flam, aper, eph)
   ...     print('{:18s} = {:5.0f}'.format(name, afrho))    # doctest: +FLOAT_CMP
   10" radius circle =  5917 cm
      5"–10" annulus = 11834 cm
         2"x10" slit = 28114 cm
   σ=5" Gaussian beam =  9442 cm


Dust syndynes and synchrones
----------------------------

Syndynes are lines in space connecting particles that are experiencing the same forces.  A syndyne is parameterized by :math:`\beta`, the ratio of the force from solar radiation to the force from solar gravity, :math:`F_r / F_g`, and age (or time of release).  Thus, all particles in a syndyne have a constant :math:`\beta` but variable age.  Similarly, synchrones are lines of constant particle age, but variable :math:`\beta`.

State objects
^^^^^^^^^^^^^

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


Syndynes
^^^^^^^^

Syndynes are generated with the `~sbpy.activity.dust.syndynes.Syndynes` class.  The class requires a `~sbpy.activity.dust.dynamics.State` object, representing the source of the syndynes, a list of :math:`\beta` values and particle ages from which to generate the syndynes.

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
^^^^^^^^^^

Synchrones are also simulated with the `~sbpy.activity.dust.syndynes.Syndynes` class, but instead generated with the :meth:`~sbpy.activity.dust.syndynes.Syndynes.get_synchrone` and :meth:`~sbpy.activity.dust.syndynes.Syndynes.synchrones` methods.

.. doctest::

   >>> age, states = syn.get_synchrone(24)
   >>> r, v = abs(states)
   >>> print(age, "{:.3g}".format(r.max().to("au")))
   8640000.0 s 2.27 AU


Projecting onto the sky
^^^^^^^^^^^^^^^^^^^^^^^

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
^^^^^^^^^^^^^^^^^^^

Calculating the positions of the projected orbit of the source object may be helpful for interpreting an observation or a set of syndynes.  They are calculated with the :meth:`~sbpy.activity.dust.synydnes.Syndynes.get_orbit` method:

.. doctest::

   >>> dt = np.linspace(-2, 2) * u.d
   >>> orbit, coords = syn.get_orbit(dt, frame="icrs")

Other dynamical models
^^^^^^^^^^^^^^^^^^^^^^

`sbpy`'s built-in models solve the equations of motion for dust grains given two-body dynamics.  Users may provide their own models in order to, e.g., improve code performance, or add planetary perturbations.  Use the `~sbpy.activity.dust.dynamics.SolarGravityAndRadiationPressure` class as a template.

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
.. automodapi:: sbpy.activity.dust
   :no-heading:
