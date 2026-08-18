#!/usr/bin/env python
# -*- coding:utf-8 -*-

# @Filename: response.py
# @License: BSD 3-clause (http://www.opensource.org/licenses/BSD-3-Clause)


import os

import astropy.io.ascii
import astropy.io.fits
import astropy.table
import astropy.units as u
import numpy as np
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.optimize as optimize
from astroquery.svo_fps import SvoFps

import kcorrect
import kcorrect.template
import kcorrect.utils


SVO_FILTER_ALIASES = {
    "alhambra_768": "CAHA/ALHAMBRA.F768W",
    "alhambra_799": "CAHA/ALHAMBRA.F799W",
    "alhambra_830": "CAHA/ALHAMBRA.F830W",
    "alhambra_861": "CAHA/ALHAMBRA.F861W",
    "alhambra_892": "CAHA/ALHAMBRA.F892W",
    "alhambra_923": "CAHA/ALHAMBRA.F923W",
    "alhambra_954": "CAHA/ALHAMBRA.F954W",
    "alhambra_H": "CAHA/ALHAMBRA.H",
    "alhambra_J": "CAHA/ALHAMBRA.J",
    "alhambra_KS": "CAHA/ALHAMBRA.Ks",
    "alhambra_f365w": "CAHA/ALHAMBRA.F365W",
    "alhambra_f396w": "CAHA/ALHAMBRA.F396W",
    "alhambra_f427w": "CAHA/ALHAMBRA.F427W",
    "alhambra_f458w": "CAHA/ALHAMBRA.F458W",
    "alhambra_f489w": "CAHA/ALHAMBRA.F489W",
    "alhambra_f520w": "CAHA/ALHAMBRA.F520W",
    "alhambra_f551w": "CAHA/ALHAMBRA.F551W",
    "alhambra_f582w": "CAHA/ALHAMBRA.F582W",
    "alhambra_f613w": "CAHA/ALHAMBRA.F613W",
    "alhambra_f644w": "CAHA/ALHAMBRA.F644W",
    "alhambra_f675w": "CAHA/ALHAMBRA.F675W",
    "alhambra_f706w": "CAHA/ALHAMBRA.F706W",
    "alhambra_f737w": "CAHA/ALHAMBRA.F737W",
    "alhambra_f768w": "CAHA/ALHAMBRA.F768W",
    "alhambra_f799w": "CAHA/ALHAMBRA.F799W",
    "alhambra_f830w": "CAHA/ALHAMBRA.F830W",
    "alhambra_f861w": "CAHA/ALHAMBRA.F861W",
    "alhambra_f892w": "CAHA/ALHAMBRA.F892W",
    "alhambra_f923w": "CAHA/ALHAMBRA.F923W",
    "alhambra_f954w": "CAHA/ALHAMBRA.F954W",
    "bass_g": "BOK/BASS.g",
    "bass_r": "BOK/BASS.r",
    "bessell_B": "Generic/Bessell.B",
    "bessell_I": "Generic/Bessell.I",
    "bessell_R": "Generic/Bessell.R",
    "bessell_U": "Generic/Bessell.U",
    "bessell_V": "Generic/Bessell.V",
    "bok_90prime_z": "BOK/90prime.sdss_z",
    "capak_cfht_megaprime_sagem_g": "CFHT/Megaprime.g",
    "capak_cfht_megaprime_sagem_i": "CFHT/Megaprime.i",
    "capak_cfht_megaprime_sagem_r": "CFHT/Megaprime.r",
    "capak_cfht_megaprime_sagem_u": "CFHT/Megaprime.u",
    "capak_cfht_megaprime_sagem_z": "CFHT/Megaprime.z",
    "capak_cfht_wircam_Ks": "CFHT/Wircam.Ks",
    "capak_subaru_suprimecam_B": "Subaru/Suprime.B",
    "capak_subaru_suprimecam_V": "Subaru/Suprime.V",
    "capak_subaru_suprimecam_g": "Subaru/Suprime.g",
    "capak_subaru_suprimecam_i": "Subaru/Suprime.i",
    "capak_subaru_suprimecam_r": "Subaru/Suprime.r",
    "capak_subaru_suprimecam_z": "Subaru/Suprime.z",
    "capak_ukirt_wfcam_J": "UKIRT/WFCAM.J",
    "capak_ukirt_wfcam_Y": "UKIRT/WFCAM.Y",
    "cdfs_swire_U": "CTIO/MosaicII.U",
    "cdfs_swire_g": "CTIO/MosaicII.gSDSS",
    "cdfs_swire_i": "CTIO/MosaicII.iSDSS",
    "cdfs_swire_r": "CTIO/MosaicII.rSDSS",
    "cdfs_swire_z": "CTIO/MosaicII.zSDSS",
    "cfh12k_B": "CFHT/CFH12k.B",
    "cfh12k_I": "CFHT/CFH12k.I",
    "cfh12k_R": "CFHT/CFH12k.R",
    "cfh12k_V": "CFHT/CFH12k.V",
    "cfh12k_z": "CFHT/CFH12k.Zprime",
    "cfht_megacam_g": "CFHT/MegaCam.g",
    "cfht_megacam_i": "CFHT/MegaCam.i",
    "cfht_megacam_r": "CFHT/MegaCam.r",
    "cfht_megacam_u": "CFHT/MegaCam.u",
    "cfht_megacam_z": "CFHT/MegaCam.z",
    "clash_acs_f435w": "HST/ACS_WFC.F435W",
    "clash_acs_f475w": "HST/ACS_WFC.F475W",
    "clash_acs_f555w": "HST/ACS_WFC.F555W",
    "clash_acs_f606w": "HST/ACS_WFC.F606W",
    "clash_acs_f625w": "HST/ACS_WFC.F625W",
    "clash_acs_f775w": "HST/ACS_WFC.F775W",
    "clash_acs_f814w": "HST/ACS_WFC.F814W",
    "clash_acs_f850lp": "HST/ACS_WFC.F850LP",
    "clash_wfc3_f105w": "HST/WFC3_IR.F105W",
    "clash_wfc3_f110w": "HST/WFC3_IR.F110W",
    "clash_wfc3_f125w": "HST/WFC3_IR.F125W",
    "clash_wfc3_f140w": "HST/WFC3_IR.F140W",
    "clash_wfc3_f160w": "HST/WFC3_IR.F160W",
    "clash_wfc3_f218w": "HST/WFC3_UVIS2.F218W",
    "clash_wfc3_f225w": "HST/WFC3_UVIS2.F225W",
    "clash_wfc3_f275w": "HST/WFC3_UVIS2.F275W",
    "clash_wfc3_f300x": "HST/WFC3_UVIS1.F300X",
    "clash_wfc3_f336w": "HST/WFC3_UVIS1.F336W",
    "clash_wfc3_f390w": "HST/WFC3_UVIS2.F390W",
    "clash_wfc3_f438w": "HST/WFC3_UVIS2.F438W",
    "clash_wfc3_f475w": "HST/WFC3_UVIS2.F475W",
    "clash_wfc3_f555w": "HST/WFC3_UVIS2.F555W",
    "clash_wfc3_f606w": "HST/WFC3_UVIS2.F606W",
    "clash_wfc3_f625w": "HST/WFC3_UVIS2.F625W",
    "clash_wfc3_f775w": "HST/WFC3_UVIS2.F775W",
    "clash_wfc3_f814w": "HST/WFC3_UVIS2.F814W",
    "clash_wfc3_f850lp": "HST/WFC3_UVIS2.F850LP",
    "ctio_mosaic_ii_B": "CTIO/MosaicII.B",
    "ctio_mosaic_ii_Ic": "CTIO/MosaicII.I",
    "ctio_mosaic_ii_Rc": "CTIO/MosaicII.R",
    "ctio_mosaic_ii_Uj": "CTIO/MosaicII.U",
    "ctio_mosaic_ii_V": "CTIO/MosaicII.V",
    "ctio_mosaic_ii_g": "CTIO/MosaicII.gSDSS",
    "ctio_mosaic_ii_i": "CTIO/MosaicII.iSDSS",
    "ctio_mosaic_ii_r": "CTIO/MosaicII.rSDSS",
    "ctio_mosaic_ii_u": "CTIO/MosaicII.uSDSS",
    "ctio_mosaic_ii_z": "CTIO/MosaicII.zSDSS",
    "decam_Y": "CTIO/DECam.Y",
    "decam_g": "CTIO/DECam.g",
    "decam_i": "CTIO/DECam.i",
    "decam_r": "CTIO/DECam.r",
    "decam_u": "CTIO/DECam.u",
    "decam_z": "CTIO/DECam.z",
    "deep_B": "CFHT/CFH12k.B",
    "deep_I": "CFHT/CFH12k.I",
    "deep_R": "CFHT/CFH12k.R",
    "ediscs_B": "Paranal/FORS1.ESO1034",
    "ediscs_I": "Paranal/FORS2.ESO1077",
    "ediscs_R": "Paranal/FORS1.ESO1036",
    "ediscs_V": "Paranal/FORS1.ESO1035",
    "epsi_420m": "LaSilla/COMBO17.420nm",
    "epsi_464m": "LaSilla/COMBO17.464nm",
    "epsi_485m": "LaSilla/COMBO17.485nm",
    "epsi_518m": "LaSilla/COMBO17.518nm",
    "epsi_571m": "LaSilla/COMBO17.571nm",
    "epsi_604m": "LaSilla/COMBO17.604nm",
    "epsi_646m": "LaSilla/COMBO17.646nm",
    "epsi_696m": "LaSilla/COMBO17.696nm",
    "epsi_753m": "LaSilla/COMBO17.753nm",
    "epsi_815m": "LaSilla/COMBO17.815nm",
    "epsi_855m": "LaSilla/COMBO17.855nm",
    "epsi_B": "LaSilla/COMBO17.B",
    "epsi_I": "LaSilla/COMBO17.I",
    "epsi_R": "LaSilla/COMBO17.R",
    "epsi_U": "LaSilla/COMBO17.U",
    "epsi_V": "LaSilla/COMBO17.V",
    "flamingos_H": "KPNO/Flamingos.H",
    "flamingos_J": "KPNO/Flamingos.J",
    "flamingos_Ks": "KPNO/Flamingos.Ks",
    "FORS_B_ccd_atm": "Paranal/FORS1.ESO1034",
    "FORS_I_ccd_atm": "Paranal/FORS1.ESO1037",
    "FORS_V_ccd_atm": "Paranal/FORS1.ESO1035",
    "FORS2_R_ccd_atm": "Paranal/FORS2.ESO1076",
    "galex_FUV": "GALEX/GALEX.FUV",
    "galex_NUV": "GALEX/GALEX.NUV",
    "gdds_B": "CTIO/MosaicII.B",
    # "gdds_H",
    "gdds_I": "CTIO/MosaicII.I",
    # "gdds_K",
    "gdds_R": "CTIO/MosaicII.R",
    "gdds_V": "CTIO/MosaicII.V",
    "gdds_z": "CTIO/MosaicII.zSDSS",
    "goods_H_isaac_etc": "Paranal/ISAAC.H",
    "goods_J_isaac_etc": "Paranal/ISAAC.Js",
    "goods_Ks_isaac_etc": "Paranal/ISAAC.Ks",
    "goods_acs_f435w": "HST/ACS_WFC.F435W",
    "goods_acs_f606w": "HST/ACS_WFC.F606W",
    "goods_acs_f775w": "HST/ACS_WFC.F775W",
    "goods_acs_f850lp": "HST/ACS_WFC.F850LP",
    "hawki_Ks1": "Paranal/HAWKI.Ks",
    "herschel_pacs_100": "Herschel/Pacs.green",
    "herschel_pacs_160": "Herschel/Pacs.red",
    "herschel_pacs_70": "Herschel/Pacs.blue",
    "herschel_spire_250": "Herschel/SPIRE.PSW",
    "herschel_spire_350": "Herschel/SPIRE.PMW",
    "herschel_spire_500": "Herschel/SPIRE.PLW",
    "herschel_spire_ext_250": "Herschel/SPIRE.PSW_ext",
    "herschel_spire_ext_350": "Herschel/SPIRE.PMW_ext",
    "herschel_spire_ext_500": "Herschel/SPIRE.PLW_ext",
    "hst_acs_f814w": "HST/ACS_WFC.F814W",
    "iras_100": "IRAS/IRAS.100mu",
    "iras_12": "IRAS/IRAS.12mu",
    "iras_25": "IRAS/IRAS.25mu",
    "iras_60": "IRAS/IRAS.60mu",
    # JWST NIRCam aliases populated below
    "jwst_f560w": "JWST/MIRI.F560W",
    "jwst_f770w": "JWST/MIRI.F770W",
    "lbc_blue_ufilter": "LBT/LBCB.bessel-U",
    "lbc_red_yfilter": "LBT/LBCR.Y",
    "lco_wirc_H": "LCO/WIRC.H",
    "lco_wirc_J": "LCO/WIRC.J",
    # "lco_wirc_Ks": "LCO/WIRC.Ks",
    "mmt_megacam_g": "MMT/MegaCam.g_MMT",
    "mmt_megacam_i": "MMT/MegaCam.i_MMT",
    "mmt_megacam_r": "MMT/MegaCam.r_MMT",
    "mmt_megacam_u": "MMT/MegaCam.u_MMT",
    "mmt_megacam_z": "MMT/MegaCam.z_MMT",
    "moircs_K": "Subaru/MOIRCS.K",
    "mzls_z": "KPNO/MzLS.z",
    # "ndwfs_Bw",
    "ndwfs_I": "KPNO/Mosaic.I",
    # "ndwfs_K",
    "ndwfs_R": "KPNO/Mosaic.U",
    "newfirm_H": "NOAO/NEWFIRM.HX",
    "newfirm_J": "NOAO/NEWFIRM.JX",
    "newfirm_Ks": "NOAO/NEWFIRM.Ks",
    "palomar_K": "P200/WIRC.Ks",
    # SDSS aliases populated below
    "SOFI_J_atm": "LaSilla/SOFI.J",
    "SOFI_Ks_atm": "LaSilla/SOFI.Ks",
    "sofia_hawc_bandA": "SOFIA/HAWC.A",
    "sofia_hawc_bandB": "SOFIA/HAWC.B",
    "sofia_hawc_bandC": "SOFIA/HAWC.C",
    "sofia_hawc_bandD": "SOFIA/HAWC.D",
    "sofia_hawc_bandE": "SOFIA/HAWC.E",
    "spitzer_irac_ch1": "Spitzer/IRAC.I1",
    "spitzer_irac_ch2": "Spitzer/IRAC.I2",
    "spitzer_irac_ch3": "Spitzer/IRAC.I3",
    "spitzer_irac_ch4": "Spitzer/IRAC.I4",
    "spitzer_mips_160": "Spitzer/MIPS.160mu",
    "spitzer_mips_24": "Spitzer/MIPS.24mu",
    "spitzer_mips_70": "Spitzer/MIPS.70mu",
    "subaru_suprimecam_B": "Subaru/Suprime.B",
    "subaru_suprimecam_Ic": "Subaru/Suprime.Ic_filter",
    "subaru_suprimecam_Rc": "Subaru/Suprime.Rc_filter",
    "subaru_suprimecam_V": "Subaru/Suprime.V",
    "subaru_suprimecam_g": "Subaru/Suprime.g",
    "subaru_suprimecam_i": "Subaru/Suprime.i",
    "subaru_suprimecam_r": "Subaru/Suprime.r",
    "subaru_suprimecam_z": "Subaru/Suprime.z",
    "twomass_J": "2MASS/2MASS.J",
    "twomass_H": "2MASS/2MASS.H",
    "twomass_Ks": "2MASS/2MASS.Ks",
    "ukirt_wfcam_Brg": "UKIRT/WFCAM.Brg_filter",
    "ukirt_wfcam_H": "UKIRT/WFCAM.H_filter",
    "ukirt_wfcam_H2": "UKIRT/WFCAM.H2_filter",
    "ukirt_wfcam_J": "UKIRT/WFCAM.J_filter",
    "ukirt_wfcam_K": "UKIRT/WFCAM.K_filter",
    "ukirt_wfcam_Y": "UKIRT/WFCAM.Y_filter",
    "ukirt_wfcam_Z": "UKIRT/WFCAM.Z_filter",
    # "ukschmidt_bj",
    "vimos_B": "Paranal/VIMOS.B",
    "vimos_I": "Paranal/VIMOS.I",
    "vimos_R": "Paranal/VIMOS.R",
    "vimos_U": "Paranal/VIMOS.U",
    "vimos_V": "Paranal/VIMOS.V",
    "vlt_vimos_I": "Paranal/VIMOS.I",
    "vlt_vimos_z": "Paranal/VIMOS.z",
    "wfcam_H": "UKIRT/WFCAM.H",
    "wfcam_J": "UKIRT/WFCAM.J",
    "wfcam_K": "UKIRT/WFCAM.K",
    "wfcam_Z": "UKIRT/WFCAM.Z",
    "wfpc2_f450w": "HST/WFPC2-PC.F450W",
    "wfpc2_f555w": "HST/WFPC2-PC.F555W",
    "wfpc2_f606w": "HST/WFPC2-PC.F606W",
    "wfpc2_f675w": "HST/WFPC2-PC.F675W",
    "wfpc2_f702w": "HST/WFPC2-PC.F702W",
    "wfpc2_f814w": "HST/WFPC2-PC.F814W",
    "wise_w1": "WISE/WISE.W1",
    "wise_w2": "WISE/WISE.W2",
    "wise_w3": "WISE/WISE.W3",
    "wise_w4": "WISE/WISE.W4",
}
for sdss_filter in "ugriz":
    for i in range(7):
        SVO_FILTER_ALIASES[f"sdss_{sdss_filter}{i:d}"] = f"SLOAN/SDSS.{sdss_filter}"
for jwst_filter in ["f070w", "f090w", "f1000w", "f1130w", "f115w", "f1280w", "f140m",
                    "f1500w", "f150w", "f162m", "f1800w", "f182m", "f200w", "f2100w",
                    "f210m", "f250m", "f2550w", "f277w", "f300m", "f335m", "f356w",
                    "f360m", "f410m", "f430m", "f444w", "f460m", "f480m"]:
    SVO_FILTER_ALIASES[f"jwst_{jwst_filter}"] = f"JWST/NIRCam.{jwst_filter.upper()}"


def all_responses(facility=None, instrument=None, wavelength_min=None,
                  wavelength_max=None):
    """List response IDs available from the SVO Filter Profile Service.

    Parameters
    ----------

    facility : str, optional
        SVO facility name. If supplied, return filters for this facility.

    instrument : str, optional
        SVO instrument name. Used only with ``facility``.

    wavelength_min, wavelength_max : float or astropy.units.Quantity, optional
        Effective wavelength range. Numeric values are interpreted as Angstroms.
        Both values are required when ``facility`` is not supplied.

    Returns
    -------

    responses : list of str
        SVO filter IDs suitable for :meth:`ResponseDict.load_response`.

    Notes
    -----

    SVO does not provide a single inexpensive query for every filter. Specify
    either a facility or an effective wavelength range.
    """
    if facility is not None:
        filters = SvoFps.get_filter_list(facility=facility,
                                         instrument=instrument)
    elif ((wavelength_min is not None) and
          (wavelength_max is not None)):
        if not isinstance(wavelength_min, u.Quantity):
            wavelength_min = wavelength_min * u.Angstrom
        if not isinstance(wavelength_max, u.Quantity):
            wavelength_max = wavelength_max * u.Angstrom
        filters = SvoFps.get_filter_index(wavelength_min, wavelength_max)
    else:
        raise ValueError("specify facility or both wavelength_min and wavelength_max")

    return list(filters["filterID"])


# Class to define a singleton
class ResponseDictSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(ResponseDictSingleton,
                                        cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class ResponseDict(dict, metaclass=ResponseDictSingleton):
    """Dictionary of all responses (singleton)"""

    def __init__(self):
        return

    def load_response(self, response=None, reload=False):
        """Load a response from the SVO Filter Profile Service.

        Parameters
        ----------

        response : str
            SVO filter ID, or one of the legacy kcorrect aliases in
            ``SVO_FILTER_ALIASES``.

        reload : bool
            if True, reload the response if already in ResponseDict (default False)
        """
        if response is None:
            raise ValueError("response must be an SVO filter ID or legacy alias")
        if response in self and not reload:
            return

        filter_id = SVO_FILTER_ALIASES.get(response)
        if filter_id is None:
            if "/" not in response:
                raise ValueError(f"unknown response {response!r}; use an SVO filter ID or a legacy alias")
            filter_id = response

        self[filter_id] = Response.from_svo(filter_id=filter_id)
        print("Loaded response for filter", filter_id)
        return


class Response(object):
    """Astronomical bandpass description

    Parameters
    ----------

    filename : str
        file name to read from

    wave : ndarray of np.float32
        wavelength grid in Angstroms

    response : ndarray of np.float32
        response function

    Attributes
    ----------

    filename : str
        source filename, or None

    svo_filter_id : str
        SVO filter ID, or None

    fwhm, fwhm_low, fwhm_high : np.float32
        FWHM of response, with low and high wavelength limits (Angstroms)

    interp : scipy.interpolate.interp1d object
        interpolation object

    lambda_eff : np.float32
        effective wavelength in Angstroms

    nwave : int
        number of wavelength samples

    response : ndarray of np.float32
        response function

    solar_magnitude : np.float32
        absolute magnitude of Sun through filter, or None

    solar_sed : kcorrect.template.SED object
        SED associated with Sun for solar_magnitude

    vega2ab : np.float32
        magnitude offset from Vega to AB (m_AB - m_Vega), or None

    vega_sed : kcorrect.template.SED object
        SED associated with Vega for vega_magnitude

    wave : ndarray of np.float32
        wavelength grid in Angstroms

    Notes
    -----

    The response should be the relative response of the system
    (atmosphere, telescope, detector, etc) to a photon at each 
    given wavelength entering the Earth's atmosphere (for a ground
    based telescope) or the telescope aperture (space based).

    If a file is given it is assumed to be in fixed_width 
    format, readable and writeable by astropy.io.ascii

    The wavelengths are sorted on input so the attribute wave is
    always increasing.

    The attribute interp() takes wavelength in Angstroms as its one
    positional argument.

    The effective wavelength is defined as described in Blanton &
    Roweis (2007).
    """

    def __init__(self, filename=None, wave=None, response=None):
        if wave is not None and response is not None:
            isort = np.argsort(wave)
            self.wave = wave[isort]
            self.response = response[isort]
        else:
            self.wave = wave
            self.response = response
        self.filename = filename
        self.svo_filter_id = None
        self.solar_sed = None
        self.solar_magnitude = None
        self.vega_sed = None
        self.vega_magnitude = None
        self.lambda_eff = None
        self.interp = None
        self.fwhm = None
        self.fwhm_low = None
        self.fwhm_high = None
        if self.filename is not None:
            self.load_dat(filename)
        else:
            self._setup()
        return

    def _setup(self):
        """Some initial setup after an input"""
        self.set_interp()
        self.set_lambda_eff()
        self.set_solar_magnitude()
        self.set_vega2ab()
        self.set_fwhm_limits()
        return

    def response_dtype(self):
        """Returns numpy dtype for SED"""
        response_dtype = np.dtype([('wave', type(self.wave[0]), self.nwave),
                                   ('response', type(self.response[0]), self.nwave)])
        return response_dtype

    def set_interp(self):
        """Sets attribute interp to interpolation function"""
        if self.wave is None or self.response is None:
            self.interp = None
            return
        self.interp = interpolate.interp1d(self.wave, self.response,
                                           kind='cubic',
                                           bounds_error=False,
                                           fill_value=0.)
        return

    @property
    def nwave(self) -> int:
        if self.wave is None:
            return 0
        return np.asanyarray(self.wave).size

    @classmethod
    def from_svo(cls, filter_id: str):
        """Read a response from the SVO Filter Profile Service.

        Parameters
        ----------
        filter_id : str
            Filter ID in the SVO form ``facility/instrument.filter``.
        """

        data = SvoFps.get_transmission_data(filter_id)
        wave = data["Wavelength"]
        if getattr(wave, "unit", None) is not None:
            wave = u.Quantity(wave).to_value(u.Angstrom)
        else:
            wave = np.asarray(wave)
        if filter_id.startswith("GALEX") and data["Transmission"].unit == u.cm**2:
            response = np.array(data["Transmission"].quantity / (1963.495*u.cm**2))
        else:
            response = data["Transmission"].quantity.value

        if len(wave) == 0:
            raise ValueError(f"SVO returned no transmission data for {filter_id}")

        instance = cls(filename=None, wave=wave, response=response)
        instance.svo_filter_id = filter_id
        return instance

    def load_fits(self, filename=None, ext=1):
        """Read response from FITS files

        Parameters
        ----------

        filename : str
            input file name

        ext : str or int
            extension to read from
        """
        response_hdulist = astropy.io.fits.open(filename)
        response = response_hdulist[ext].data[0]
        isort = np.argsort(response['wave'])
        self.wave = response['wave'][isort]
        self.response = response['response'][isort]
        self.filename = filename
        self.svo_filter_id = None
        self._setup()
        return

    def load_dat(self, filename=None):
        """Read response from fixed_width file

        Parameters
        ----------
        filename : str
            input file name
        """
        infilename = os.fspath(filename)

        if not os.path.exists(infilename):
            raise ValueError("No response file: {f}".format(f=infilename))
        dat = astropy.io.ascii.read(infilename, format='fixed_width')
        isort = np.argsort(dat['lambda'])
        self.wave = dat['lambda'][isort]
        self.response = dat['pass'][isort]
        self.filename = infilename
        self.svo_filter_id = None
        self._setup()
        return

    def tofits(self, filename=None, ext='FLUX', clobber=True):
        """Write SED to FITS files

        filename : str
            output file name

        ext : str or int
            extension to write to
        
        clobber : bool
            whether to clobber the existing file or add an HDU
        """
        out = np.zeros(1, self.response_dtype())
        out['wave'] = self.wave
        out['response'] = self.response

        out_table = astropy.table.Table(out)

        out_table.write(filename, overwrite=clobber)
        return

    def project(self, sed=None, wave=None, flux=None):
        """Project spectrum onto response

        Parameters
        ----------

        sed : kcorrect.template.SED object
            spectrum in kcorrect format

        wave : ndarray of np.float32
            wavelength grid (used only if sed and func not set)

        flux : ndarray of np.float32
            flux grid (used only if sed and func not set)

        Returns
        -------

        maggies : np.float32
            [nsed] nmaggies associated with spectrum through bandpass

        Notes
        -----

        Fluxes should be in erg cm^{-2} s^{-1} Angstrom^{-1}

        If "flux" and "wave" are specified, then wave must be
        a 1-dimensional array, and flux must be a 1-dimensional
        or 2-dimensional array, with the last axis corresponding
        to wavelength and with the same number of elements as
        wave.

        Assumes AB calibration.

        If the bandpass is outside the range of the solar model, 0 is returned.
        """
        if sed is None:
            if wave is None or flux is None:
                raise ValueError("must specify sed, or wave and flux")
            wave = np.float32(wave)
            flux = np.float32(flux)
            if wave.ndim != 1:
                raise ValueError("wave must be 1-D array")
            if flux.shape[-1] != wave.shape[0]:
                raise ValueError("last axis of flux must match wave")
            if flux.ndim > 2:
                raise ValueError("flux must be 1-D or 2-D array")
            sed_wave = wave
            if flux.ndim == 1:
                nsed = 1
            else:
                nsed = flux.shape[0]
            interp = interpolate.interp1d(wave, flux, kind='cubic',
                                          bounds_error=False, fill_value=0.)
        else:
            sed_wave = sed.wave
            interp = sed.interp
            nsed = sed.nsed

        # Find SED wavelengths to integrate over
        keep = (sed_wave >= self.wave[0]) & (sed_wave <= self.wave[-1])
        ikeep = np.where(keep)[0]
        if len(ikeep) == 0:
            return 0.

        if ikeep[0] > 0:
            keep[ikeep[0] - 1] = 1
        if ikeep[-1] < len(sed_wave) - 1:
            keep[ikeep[-1] + 1] = 1

        # Find full grid of wavelengths for integration
        integrate_wave = np.unique(np.append(sed_wave[keep], self.wave))

        # Interpolate to grid
        integrate_sed = interp(integrate_wave)
        integrate_response = self.interp(integrate_wave)

        # Perform integration for numerator
        numer = np.zeros(nsed, dtype=np.float32)
        if nsed == 1:
            integrand_numer = integrate_sed * integrate_response * integrate_wave
            numer = integrate.trapezoid(integrate_wave, np.squeeze(integrand_numer))
        else:
            for ised in np.arange(nsed, dtype=int):
                integrand_numer = integrate_sed[ised, :] * integrate_response * integrate_wave
                numer[ised] = integrate.trapezoid(integrate_wave,
                                                  np.squeeze(integrand_numer))

        # Perform integration for denominator
        integrand_denom = (kcorrect.utils.sed_ab(integrate_wave) *
                           integrate_response * integrate_wave)
        denom = integrate.trapezoid(integrate_wave, integrand_denom)

        # AB maggies are projection of SED onto response divided by same
        # projection for the AB source.
        maggies = np.squeeze(numer / denom)

        return maggies

    def set_lambda_eff(self):
        """Set effective wavelength

        Notes
        -----

        Sets attribute lambda_eff
        """
        # Just use original grid; good enough.
        wave = self.wave
        response = self.response

        # Perform integration for numerator
        integrand_numer = np.log(wave) * response / wave
        numer = integrate.trapezoid(wave, integrand_numer)

        # Perform integration for denominator
        integrand_denom = response / wave
        denom = integrate.trapezoid(wave, integrand_denom)

        # Set effective wavelength
        self.lambda_eff = np.exp(numer / denom)

        return

    def set_fwhm_limits(self):
        """Set limits for FWHM

        Notes
        -----

        Sets attributes fwhm_low, fwhm_high, fwhm.

        fwhm_low is the lowest wavelength value for which the response
        reaches 50% maximum when starting from the low end.

        fwhm_high is the highest wavelength value for which the response
        reaches 50% maximum when starting from the high end.

        fwhm is (fwhm_high - fwhm_low)
        """
        # Just use original grid; good enough.
        wave = self.wave.copy()
        iresponse = self.interp(wave)
        maxresponse = iresponse.max()
        iresponse = iresponse / maxresponse

        # Find lower
        iupper = np.where(iresponse > 0.5)[0][0]
        if iupper == 0:
            fwhm_low = wave[iupper]
        else:
            ilower = iupper - 1
            fwhm_low = optimize.brentq(lambda x : (self.interp(x) / maxresponse - 0.5),
                                       wave[ilower], wave[iupper])

        # Find upper
        ilower = np.where(iresponse >= 0.5)[0][-1]
        if ilower == len(iresponse) - 1:
            fwhm_high = wave[-1]
        else:
            iupper = ilower + 1
            fwhm_high = optimize.brentq(lambda x : (self.interp(x) / maxresponse - 0.5),
                                        wave[ilower], wave[iupper])

        self.fwhm_low = fwhm_low
        self.fwhm_high = fwhm_high
        self.fwhm = fwhm_high - fwhm_low
        return

    def set_solar_magnitude(self):
        """Set absolute magnitude of Sun through filter

        Notes
        -----

        Uses lcbsun.ori model from Lejeune et al (1997)

        If the response function is outside the model wavelength range,
        solar_magnitude is set to None.
        """
        if self.solar_sed is None:
            sunfile = os.path.join(kcorrect.KCORRECT_DIR, 'data', 'basel',
                                   'lcbsun.ori')
            info, wave, flux = kcorrect.utils.read_basel(filename=sunfile)

            # Now convert to Angstroms and erg/cm^2/s/Ang at 10 pc
            radius = 6.960e+10
            wave = wave * 10.  # nm to Angstrom
            pctocm = 3.086e+18
            cspeed = 2.99792e+18   # Ang/s
            for unit in range(flux.shape[0]):
                flux[unit, :] = np.pi * 4. * flux * cspeed / wave**2
                flux = flux * (radius / (10. * pctocm))**2
            self.solar_sed = kcorrect.template.SED(wave=wave, flux=flux)
            self.solar_sed.info = info

        solar_maggies = self.project(sed=self.solar_sed)

        if solar_maggies > 0:
            self.solar_magnitude = - 2.5 * np.log10(solar_maggies)
        else:
            self.solar_magnitude = None

        return

    def set_vega2ab(self):
        """Set Vega to AB magnitude conversion

        Notes
        -----

        Uses lcbvega.ori model from Lejeune et al (1997)

        If the response function is outside the model wavelength range,
        vega2ab is set to None.
        """
        if self.vega_sed is None:
            vegafile = os.path.join(kcorrect.KCORRECT_DIR, 'data', 'basel',
                                    'lcbvega.ori')
            info, wave, flux = kcorrect.utils.read_basel(filename=vegafile)

            # Conversion to match Hayes et al. 1985
            radius = 1.91144e+11  # Backed out to get normalization right
            dvega = 7.68  # Vega is 7.68 pc
            wave = wave * 10.  # nm to Angstrom
            pctocm = 3.086e+18
            cspeed = 2.99792e+18   # Ang/s
            for unit in range(flux.shape[0]):
                flux[unit, :] = np.pi * 4. * flux * cspeed / wave**2
                flux = flux * (radius / (dvega * pctocm))**2
            self.vega_sed = kcorrect.template.SED(wave=wave, flux=flux)
            self.vega_sed.info = info

        vega_maggies = self.project(sed=self.vega_sed)
        if vega_maggies > 0:
            self.vega2ab = - 2.5 * np.log10(vega_maggies)
        else:
            self.vega2ab = None
        return
