# -*- coding: utf-8 -*-
"""Сопоставление методов контроля дислокаций (7.1)–(7.3) [KKA56]."""

import math

import pytest

from mesa_diode.simulator import datafile
from mesa_diode.simulator import defects as d


def test_burgers_vector_and_kka_formulas():
    assert d.B_GE_CM * 1e8 == pytest.approx(4.0, abs=0.01)                 # a/√2
    assert d.true_width(20.0, 8.0) == pytest.approx(math.sqrt(20 ** 2 - 8 ** 2))
    assert math.isnan(d.true_width(10.0, 12.0))
    # (7.2): 20″ без поправок → (20″ в рад)²/(9b²) ≈ 6.5·10⁵ см⁻²
    assert d.density_from_width(20.0) == pytest.approx((20 * d.ARCSEC) ** 2 / (9 * d.B_GE_CM ** 2))
    assert d.density_from_width(20.0) == pytest.approx(6.5e5, rel=0.01)
    assert d.width_for_density(d.density_from_width(33.0)) == pytest.approx(33.0)
    assert d.EPD_TO_XRD_111 == pytest.approx(1 / 3, rel=0.02)              # [KKA56, с. 1288]


def test_detection_limit_and_corrections():
    assert d.detection_limit(15.6) == pytest.approx(5.3e4, rel=0.05)
    raw = d.xrd_density(20.0)
    assert raw.upper_bound and "верхняя граница" in raw.notes[0]
    corrected = d.xrd_density(20.0, 0.0, 15.6)
    assert not corrected.upper_bound and corrected.density < raw.density / 2
    assert math.isnan(d.xrd_density(14.0, 15.6).density)


def test_compare_explains_low_etch_pit_ratio():
    s = d.DefectSample("x", epd=1e4, afm_density=3e5, afm_rms=0.5, xrd_fwhm=20.0)
    notes = " ".join(d.compare(s)["notes"])
    assert "верхняя граница" in notes and "ниже него" in notes and "АСМ" in notes
    good = d.DefectSample("y", epd=5e5, xrd_fwhm=math.hypot(d.width_for_density(8e5), 10.0),
                          xrd_instrument=10.0)
    assert "В пределах [KKA56]" in " ".join(d.compare(good)["notes"])


def test_calibration_recovers_power_law_and_skips_upper_bounds():
    samples = []
    for i, n in enumerate((1e5, 1e6, 1e7)):
        fwhm = math.hypot(d.width_for_density(n), 10.0)
        samples.append(d.DefectSample(f"s{i}", epd=0.5 * n, xrd_fwhm=fwhm, xrd_instrument=10.0))
    samples.append(d.DefectSample("raw", epd=1e3, xrd_fwhm=20.0))          # без поправок — не входит
    c = d.calibrate(samples)
    assert c.n == 3 and c.slope == pytest.approx(1.0, abs=1e-6) and c.ratio == pytest.approx(0.5)
    assert c.predict(2e6) == pytest.approx(1e6, rel=1e-6)
    one = d.calibrate(samples[:1])
    assert one.n == 1 and one.predict(4e5) == pytest.approx(2e5)
    assert d.calibrate([d.DefectSample("e")]) is None


def test_example_table_and_lifetimes():
    samples = d.samples_from_table(datafile.EXAMPLES_DIR / "defects_example.csv")
    assert len(samples) == 6 and all(s.epd and s.xrd_fwhm for s in samples)
    c = d.calibrate(samples)
    assert c.n >= 4 and 0.7 < c.slope < 1.3
    tau = d.lifetime_by_method(samples[0], 5.5e-4)
    assert set(tau) == {d.METHOD_EPD, d.METHOD_XRD, d.METHOD_AFM}
    assert tau[d.METHOD_EPD] == pytest.approx(1 / (5.5e-4 * samples[0].epd))


def test_table_without_known_columns_is_error(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a;b\n1;2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="EPD или FWHM"):
        d.samples_from_table(path)


def test_sample_from_preset_params():
    s = d.sample_from_params("p", {"N_dis": 8e3, "afm_defects": 2e5, "xrd_fwhm": 20.0, "xrd_instrument": 5.0})
    assert s.epd == 8e3 and s.xrd.instrument == 5.0
    assert d.sample_from_params("q", {"N_dis": 0.0}).epd is None
