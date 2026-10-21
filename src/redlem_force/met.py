"""Derived URMA bands (unchanged from urma_download_v2.py)."""

from __future__ import annotations


def sat_vapor_pressure_fnc(temperature):
    """Saturation vapor pressure from temperature, FAO-56 Eq. 7. kPa."""
    return temperature.add(237.3).pow(-1).multiply(temperature) \
        .multiply(17.27).exp().multiply(0.6108)


def vpd_fnc(es, ea):
    """Vapor pressure deficit, floored at zero. kPa."""
    return es.subtract(ea).max(0)


def actual_vapor_pressure_fnc(q, pair):
    """Actual vapor pressure from specific humidity and air pressure. kPa."""
    return q.multiply(0.378).add(0.622).pow(-1).multiply(q).multiply(pair)


def prep_image(img):
    """Add derived bands vs2m, ea, vpd. Wind is adjusted 10 m -> 2 m
    (FAO-56 Eq. 47)."""
    zw = 10
    wind = img.select('WIND')
    u2 = wind.expression(
        'uz * 4.87 / log(67.8 * zw - 5.42)', {'uz': wind, 'zw': zw}
    ).rename('vs2m')
    img = img.addBands(u2)

    esat = sat_vapor_pressure_fnc(img.select('TMAX')).add(
        sat_vapor_pressure_fnc(img.select('TMIN'))).multiply(0.5)

    ea = actual_vapor_pressure_fnc(
        img.select('SPFH'), img.select('PRES').divide(1000)).rename('ea')
    img = img.addBands(ea)

    img = img.addBands(vpd_fnc(esat, img.select('ea')).rename('vpd'))
    return img
