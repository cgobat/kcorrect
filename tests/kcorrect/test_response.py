import pytest
import os
import numpy as np
from astropy.table import Table
import kcorrect
import kcorrect.template
import kcorrect.response


def test_init_response():
    """Test initialization of response"""
    nwave = 1000
    wave = np.exp(np.log(4000.) + (np.log(6000.) - np.log(4000.)) *
                  (np.arange(nwave, dtype=np.float32) + 0.5) /
                  np.float32(nwave))
    response = np.exp(- 0.5 * (wave - 5000.)**2 / (500.)**2)
    f = kcorrect.response.Response(wave=wave, response=response)
    assert f.nwave == nwave
    assert len(f.wave) == nwave
    assert len(f.response) == nwave
    return


def test_load_response():
    """Test loading of response into ResponseDict"""
    f = kcorrect.response.ResponseDict()

    # Test the load
    f.load_response('sdss_u0')
    assert(type(f['sdss_u0']) == kcorrect.response.Response)
    assert f['sdss_u0'].svo_filter_id == "SLOAN/SDSS.u"

    f.load_response("SLOAN/SDSS.g")
    assert f["SLOAN/SDSS.g"].svo_filter_id == "SLOAN/SDSS.g"

    with pytest.raises(ValueError, match="use an SVO filter ID"):
        f.load_response("obsolete_local_response")

    # Test the singleton nature of the class
    f = 0
    f = kcorrect.response.ResponseDict()
    assert(type(f['sdss_u0']) == kcorrect.response.Response)

    return


def test_fits_response():
    """Test exporting and importing response in FITS"""
    f = kcorrect.response.ResponseDict()

    # Test the load
    f.load_response('sdss_u0')

    f['sdss_u0'].tofits('tmp-response-write-and-read.fits')

    r = kcorrect.response.Response()
    r.fromfits('tmp-response-write-and-read.fits')

    assert(np.all(r.nwave == f['sdss_u0'].nwave))
    assert(np.all(r.wave == f['sdss_u0'].wave))
    assert(np.all(r.response == f['sdss_u0'].response))

    os.remove('tmp-response-write-and-read.fits')

    return


def test_all_responses(monkeypatch):
    """Test listing response IDs from SVO."""
    expected = ["SLOAN/SDSS.u", "SLOAN/SDSS.g"]

    def get_filter_list(facility, instrument=None):
        assert facility == "SLOAN"
        assert instrument == "SDSS"
        return Table({"filterID": expected})

    monkeypatch.setattr(kcorrect.response.SvoFps, "get_filter_list", get_filter_list)
    responses = kcorrect.response.all_responses(facility="SLOAN", instrument="SDSS")
    assert responses == expected
    return


def test_ab_mag_simple():
    """Test that AB source gets magnitude 0"""
    f = kcorrect.response.ResponseDict()
    f.load_response('sdss_u0')

    nwave = 10000
    wave = np.exp(np.log(3000.) + (np.log(6000.) - np.log(3000.)) *
                  (np.arange(nwave, dtype=np.float32) + 0.5) /
                  np.float32(nwave))
    flux = 3631e-23 * 2.99792e+18 / wave**2
    s = kcorrect.template.SED(wave=wave, flux=flux)

    maggies = f['sdss_u0'].project(sed=s)

    assert np.abs(maggies - 1.) < 1.e-7

    return


def test_vega2ab():
    """Test that Vega-to-AB conversion is about right"""
    f = kcorrect.response.ResponseDict()
    f.load_response('sdss_u0')
    f.load_response('twomass_Ks')

    assert hasattr(f['sdss_u0'], 'vega2ab')
    assert np.isfinite(f['sdss_u0'].vega2ab)

    assert hasattr(f['twomass_Ks'], 'vega2ab')
    assert np.isfinite(f['twomass_Ks'].vega2ab)

    return


def test_solar_magnitudes():
    """Test that solar absolute magnitudes is about right"""
    f = kcorrect.response.ResponseDict()
    f.load_response('sdss_u0')
    f.load_response('twomass_Ks')

    assert hasattr(f['sdss_u0'], 'solar_magnitude')
    assert np.isfinite(f['sdss_u0'].solar_magnitude)

    assert hasattr(f['twomass_Ks'], 'solar_magnitude')
    assert np.isfinite(f['twomass_Ks'].solar_magnitude)

    return
