# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Spectroscopy Module

created on June 23, 2017
"""

from abc import ABC
import numpy as np
import astropy.units as u
import astropy.constants as const

__all__ = ['Spectrum', 'SpectralModel']


def einstein_coeff(frequency):
    """
    Einstein coefficient from molecular data

    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
        Transition frequency

    Returns
    -------
    einstein_coeff : float
        Spontaneous emission coefficient

    not implemented
    """


def total_number(integrated_flux, frequency):
    """
    Basic equation relating number of molecules with observed integrated flux.
    This is given by equation 10 in
    https://ui.adsabs.harvard.edu/#abs/2004come.book..523C

    Parameters
    ----------
    integrated_flux : `~astropy.units.Quantity`
        Integrated flux of emission line.
    frequency : `~astropy.units.Quantity`
        Transition frequency

    Returns
    -------
    total_number : float
        Total number of molecules within the aperture

    not implemented
    """
    total_number = integrated_flux
    total_number *= 8*np.pi*u.k_B*frequency**2/(const.h*const.c**3 *
                                                einstein_coeff(frequency))
    return total_number


class SpectralModel():
    """Range of spectral models"""
    def haser():
        """Haser model

        should allow direct creation of a `sbpy.actvity.Haser` instance"""
        pass

    def emission_lines():
        """Emission lines"""
        pass

    def reflectance():
        """Reflectance spectrum (asteroids)"""


class Spectrum():

    def __init__(self, flux, dispersionaxis, unit):
        self.flux = flux
        self.axis = dispersionaxis
        self.unit = unit

    @classmethod
    def read(cls, filename, columns='auto'):
        """Read spectrum from file

        Parameters
        ----------
        filename : str, mandatory
            data file name
        columns : str or list-like, optional, default: 'auto'
            file format, `auto` will try to recognize the format
            automatically

        Returns
        -------
        `Spectrum` instance

        Examples
        --------
        >>> spec = Spectrum.read('2012_XY.dat') # doctest: +SKIP

        not yet implemented

        """

    def write(self, filename, columns='all'):
        """Write spectrum to file

        Parameters
        ----------
        filename : str, mandatory
            data file name
        columns : str or list-like, optional: default: 'all'
            file format; `all` will write all fields to the file

        Examples
        --------
        >>> spec = Spectrum.read('2012_XY.dat') # doctest: +SKIP
        >>> spec.write('2012_XY.dat.bak') # doctest: +SKIP

        not yet implemented

        """

    def convert_units(self, **kwargs):
        """Convert Spectrum units as provided by user

        Examples
        --------
        >>> spec.convert_units(flux_unit=u.K) # doctest: +SKIP
        >>> spec.convert_units(dispersion_unit=u.km/u.s) # doctest: +SKIP

        not yet implemented

        """

    def baseline(self, subtract=False):
        """fit baseline to `Spectrum` instance

        Parameters
        ----------
        subtract : bool, optional, default=False
            if `True`, subtract the baseline

        Returns
        -------
        float

        Examples
        --------
        >>> baseline = spec.baseline() # doctest: +SKIP
        >>> spec.baseline(subtract=True) # doctest: +SKIP

        not yet implemented

        """

    def slope(self, subtract=False):
        """fit slope to `Spectrum` instance

        Parameters
        ----------
        subtract : bool, optional, default=False
            if `True`, subtract the slope

        Returns
        -------
        float

        Examples
        --------
        >>> slope = spec.slope() # doctest: +SKIP
        >>> spec.slope(subtract=True) # doctest: +SKIP

        not yet implemented

        """

    def integrated_flux(self, frequency, interval=1*u.km/u.s):
        """
        Calculate integrated flux of emission line.

        Parameters
        ----------
        frequency : `~astropy.units.Quantity`
            Transition frequency
        interval : `~astropy.units.Quantity`
            line width

        Examples
        --------
        >>> flux = spec.integrated_flux(frequency=556.9*u.GHz, # doctest: +SKIP
        >>>                             interval=1.7*u.km/u.s) # doctest: +SKIP

        not yet implemented

        """

    def fit(self, spec):
        """Fit `SpectralModel` to different model types

        Parameters
        ----------
        spec : str, mandatory
            `SpectralModel` instance to fit

        Examples
        --------
        >>> spec_model = SpectralModel(type='Haser', molecule='H2O')        # doctest: +SKIP

        >>> spec.fit(spec_model) # doctest: +SKIP
        >>> print(spec.fit_info) # doctest: +SKIP

        not yet implemented

        """

    def production_rate(self, coma, molecule, frequency, aper):
        """
        Calculate production rate for `GasComa`

        Parameters
        ----------
        coma : `sbpy.activity.gas.GasComa`
            Gas coma model

        Returns
        -------
        Q : `~astropy.units.Quantity`
            production rate

        Examples
        --------
        >>> from sbpy.activity.gas import Haser
        >>> coma = Haser(Q, v, parent) # doctest: +SKIP
        >>> Q = spec.production_rate(coma, molecule='H2O') # doctest: +SKIP

        not yet implemented

        """

        from ..activity.gas import GasComa

        if not isinstance(coma, GasComa):
            raise ValueError('coma must be a GasComa instance.')

        integrated_line = self.integrated_flux(frequency)

        molecules = total_number(integrated_line, frequency)

        model_molecules = coma.total_number(aper)

        Q = coma.q * molecules/model_molecules

        return Q

    def plot(self):
        """Plot spectrum

        Returns
        -------
        `matplotlib.pyplot` instance

        not yet implemented
        """


class SpectralStandard(ABC):
    """Abstract base class for SBPy spectral standards.

    Parameters
    ----------
    source : `~synphot.SourceSpectrum`
      The source spectrum.

    description : string, optional
      A brief description of the source spectrum.

    bibcode : string, optional
      Bibliography code for `sbpy.bib.register`.

    Attributes
    ----------
    wave        - Wavelengths of the source spectrum.
    fluxd       - Source spectrum.
    description - Brief description of the source spectrum.
    meta        - Meta data from `source`.

    """

    def __init__(self, source, description=None, bibcode=None):
        self._source = source
        self._description = description
        self._bibcode = bibcode

    @classmethod
    def from_array(cls, wave, fluxd, meta=None, **kwargs):
        """Create standard from arrays.

        Parameters
        ----------
        wave : `~astropy.units.Quantity`
          The spectral wavelengths.

        fluxd : `~astropy.units.Quantity`
          The solar flux densities, at 1 au.

        meta : dict, optional
          Meta data.

        **kwargs
          Passed to object initialization.

        """

        import synphot

        source = synphot.SourceSpectrum(
            synphot.Empirical1D, points=wave, lookup_table=fluxd,
            meta=meta)

        return cls(source, **kwargs)

    @classmethod
    def from_file(cls, filename, wave_unit=None, flux_unit=None,
                  cache=True, **kwargs):
        """Load the source spectrum from a file.

        Parameters
        ----------
        filename : string
          The name of the file.  See
          `~synphot.SourceSpectrum.from_file` for details.

        wave_unit, flux_unit : str or `~astropy.units.core.Unit`, optional
          Wavelength and flux units.

        cache : bool, optional
          If `True`, cache the contents of URLs.

        **kwargs
          Passed to `Sun` initialization.

        """

        from astropy.utils.data import download_file
        from astropy.utils.data import _is_url
        import synphot
        from synphot.specio import read_fits_spec, read_ascii_spec

        # URL cache because synphot.SourceSpectrum.from_file does not
        if _is_url(filename):
            if filename.lower().endswith(('.fits', '.fit')):
                read_spec = read_fits_spec
            else:
                read_spec = read_ascii_spec

            fn = download_file(filename, cache=True)
            spec = read_spec(fn, wave_unit=wave_unit, flux_unit=flux_unit)
            source = synphot.SourceSpectrum(
                synphot.Empirical1D, points=spec[1], lookup_table=spec[2],
                meta={'header': spec[0]})
        else:
            source = synphot.SourceSpectrum.from_file(
                filename, wave_unit=wave_unit, flux_unit=flux_unit)

        return cls(source, **kwargs)

    @property
    def description(self):
        """Description of the source spectrum."""
        return self._description

    @property
    def wave(self):
        """Wavelengths of the source spectrum."""
        return self.source.waveset

    @property
    def fluxd(self):
        """The source spectrum."""
        return self.source(self._source.waveset, flux_unit='W / (m2 um)')

    @property
    def source(self):
        from .. import bib
        if self._bibcode is not None:
            bib.register('spectroscopy', {self._description, self._bibcode})
        return self._source

    @property
    def meta(self):
        self._source.meta

    def __call__(self, wave_or_freq, unit=None):
        """Evaluate the source spectrum.

        Parameters
        ----------
        wave_or_freq : `~astropy.units.Quantity`
          Requested wavelength or frequencies of the resulting
          spectrum.  If an array, `wave_or_freq` specifies bin
          centers.  If a single value, fluxd will be interpolated to
          this wavelength/frequency.

        unit : string, `~astropy.units.Unit`, optional
          Spectral units of the output: flux density, 'vegamag',
          'ABmag', or 'STmag'.  If ``None``, return units are W/(m2
          μm) for ``wave_or_freq`` as wavelength, otherwise return Jy.

        Returns
        -------
        fluxd : `~astropy.units.Quantity`
          The spectrum binned to match `wave_or_freq`.  If a single
          point is requested, the original spectrum is interpolated to
          it.

        """

        import numpy as np
        import synphot

        if unit is None:
            if wave_or_freq.unit.is_equivalent('m'):
                unit = u.Unit('W/(m2 um)')
            else:
                unit = u.Jy

        if np.size(wave_or_freq) > 1:
            # Method adapted from http://www.astrobetter.com/blog/2013/08/12/python-tip-re-sampling-spectra-with-pysynphot/
            specele = synphot.SpectralElement(synphot.ConstFlux1D(1))
            obs = synphot.Observation(self.source, specele, binset=wave_or_freq,
                                      force='taper')

            # The following is the same as obs.binflux, except sample_binned
            # will do the unit coversions.
            fluxd = obs.sample_binned(flux_unit=unit)
        else:
            fluxd = self.source(wave_or_freq, unit)

        return fluxd

    def filt(self, bp, unit='W / (m2 um)', **kwargs):
        """Spectrum observed through a filter.

        Parameters
        ----------
        bp : string or `~synphot.SpectralElement`
          The name of a filter, or a transmission spectrum as a
          `~synphot.SpectralElement`.  See notes for built-in filter
          names.

        unit : string, `~astropy.units.Unit`, optional
          Spectral units of the output: flux density, 'vegamag',
          'ABmag', or 'STmag'.  See :ref:`sbpy_spectral_standards`
          for calibration notes.

        **kwargs
          Additional keyword arguments for
          `~synphot.observation.Observation`.

        Returns
        -------
        wave : `~astropy.units.Quantity`
          Effective wavelength.

        fluxd : `~astropy.units.Quantity` or float
          Flux density or magnitude.


        Notes
        -----

        Filter reference data is from STScI's Calibration Reference
        Data System.

        * ``'bessel_j'`` (Bessel *J*)
        * ``'bessel_h'`` (Bessel *H*)
        * ``'bessel_k'`` (Bessel *K*)
        * ``'cousins_r'`` (Cousins *R*)
        * ``'cousins_i'`` (Cousins *I*)
        * ``'johnson_u'`` (Johnson *U*)
        * ``'johnson_b'`` (Johnson *B*)
        * ``'johnson_v'`` (Johnson *V*)
        * ``'johnson_r'`` (Johnson *R*)
        * ``'johnson_i'`` (Johnson *I*)
        * ``'johnson_j'`` (Johnson *J*)
        * ``'johnson_k'`` (Johnson *K*)

        """

        import synphot
        from synphot.units import VEGAMAG
        from .vega import Vega

        if isinstance(bp, str):
            bp = synphot.SpectralElement.from_filter(bp)

        if not isinstance(bp, synphot.SpectralElement):
            raise ValueError(
                '`bp` must be a string (filter name) or `synphot.SpectralElement`.')

        obs = synphot.Observation(self.source, bp, **kwargs)
        wave = obs.effective_wavelength()

        if str(unit).lower().strip() == 'vegamag':
            f = obs.effstim('W/(m2 um)')
            f0 = Vega.from_default().filt(bp, unit='W/(m2 um)')[1]
            fluxd = -2.5 * np.log10((f / f0).value) * VEGAMAG
        else:
            fluxd = obs.effstim(unit)

        return wave, fluxd
