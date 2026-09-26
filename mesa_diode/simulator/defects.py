# -*- coding: utf-8 -*-
"""Сопоставление методов контроля дислокаций: травление (EPD), АСМ, XRD.

Разрушающий метод — подсчёт ямок травления (EPD, см⁻²) — сравнивается с
неразрушающими: полушириной кривой качания рентгеновской дифракции
(Δω₁/₂) и данными АСМ (плотность поверхностных дефектов, шероховатость RMS).
Цель — откалибровать неразрушающий контроль по травлению и дальше оценивать
плотность дислокаций новых образцов без их разрушения.

Модель связи XRD ↔ плотность дислокаций [KKA56, с. 1287]:
  (7.1) K² = β² − φ²   — истинная полуширина K по измеренной β за вычетом
        ширины первого кристалла (прибора) φ; та же поправка — на
        естественную (дарвиновскую) ширину отражения;
  (7.2) N_D = K²/(9b²) — плотность дислокаций (по Гею–Хиршу–Келли), b —
        межатомное расстояние в плоскости скольжения; для Ge b = a/√2 = 4.0 Å
        (вектор Бюргерса 60°-дислокации a/2⟨011⟩ [Giunto24, с. 041333-17]).

Соотношение методов по [KKA56, с. 1288]: EPD ≤ N_XRD — ямки не образуют
дислокации, наклонённые к поверхности больше чем на ~30°, близкие ямки
сливаются, винтовые сегменты ямок не дают, но уширяют кривую; для
поверхности (111) и дислокаций {211} геометрия даёт EPD/N_XRD ≈ 1/3;
измеренное соотношение лежит между 1 и 1/3 при 10⁴–10⁷ см⁻².

Предел чувствительности XRD (7.3) — оценка этого модуля: при точности
отсчёта δβ и поправочной ширине φ различимое уширение K_min ≈ √(2φ·δβ)
(β = φ + δβ в (7.1)), отсюда N_min = K_min²/(9b²).

Связь АСМ и RMS с дислокациями в источниках проекта не установлена: для них
модуль даёт только отношения к EPD и эмпирическую калибровку по нескольким
образцам (степенная зависимость).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mesa_diode.simulator import physics as ph
from mesa_diode.simulator.materials import GE

ARCSEC = math.pi / (180.0 * 3600.0)          # рад в угловой секунде
B_GE_CM = GE.a_angstrom / math.sqrt(2.0) * 1e-8   # b = a/√2 = 4.0 Å [Giunto24], см
DARWIN_GE_111_ARCSEC = 15.6                    # Ge (111), Cu Kα₁ [KKA56, с. 1286]
KKA_ACCURACY_ARCSEC = 1.0                      # точность отсчёта угла [KKA56, с. 1286]
EPD_TO_XRD_111 = 3.0 / (3.0 + 6.0 * 0.85 + 3.0 * 0.33)   # ≈ 0.33 [KKA56, с. 1288]
KKA_RANGE = (1e4, 1e7)                         # см⁻², где методы согласовались [KKA56]

METHOD_EPD, METHOD_XRD, METHOD_AFM = "epd", "xrd", "afm"
METHOD_LABELS = {METHOD_EPD: "ямки травления (EPD)", METHOD_XRD: "XRD, Δω₁/₂ по (7.2)",
                 METHOD_AFM: "АСМ, поверхностные дефекты"}


# ------------------------------------------------------------ XRD
def true_width(beta_arcsec, *corrections_arcsec):
    """(7.1) K = √(β² − Σφ²), угл. с; nan, если поправки не меньше β."""
    k2 = beta_arcsec ** 2 - sum(c ** 2 for c in corrections_arcsec if c)
    return math.sqrt(k2) if k2 > 0 else float("nan")


def density_from_width(K_arcsec, b_cm=B_GE_CM):
    """(7.2) N_D = K²/(9b²), см⁻²; K — истинная полуширина, угл. с."""
    K = K_arcsec * ARCSEC
    return K * K / (9.0 * b_cm * b_cm)


def width_for_density(N_cm2, b_cm=B_GE_CM):
    """Обратная (7.2): полуширина K, угл. с, которую дала бы плотность N."""
    return 3.0 * b_cm * math.sqrt(N_cm2) / ARCSEC


def detection_limit(phi_arcsec, accuracy_arcsec=KKA_ACCURACY_ARCSEC, b_cm=B_GE_CM):
    """(7.3) Наименьшая различимая плотность, см⁻² (оценка): K_min = √(2φδβ + δβ²)."""
    K = math.sqrt(2.0 * phi_arcsec * accuracy_arcsec + accuracy_arcsec ** 2)
    return density_from_width(K, b_cm)


@dataclass
class XrdEstimate:
    fwhm: float                 # измеренная β, угл. с
    instrument: float           # φ прибора (первого кристалла), угл. с
    intrinsic: float            # естественная ширина отражения, угл. с
    K: float                    # истинная полуширина, угл. с
    density: float              # N_D по (7.2), см⁻²
    upper_bound: bool           # поправки не заданы — N_D только верхняя граница
    limit: float                # предел чувствительности (7.3), см⁻²
    notes: list = field(default_factory=list)


def xrd_density(fwhm_arcsec, instrument_arcsec=0.0, intrinsic_arcsec=0.0, b_cm=B_GE_CM,
                accuracy_arcsec=KKA_ACCURACY_ARCSEC):
    """Плотность дислокаций по полуширине кривой качания с поправками (7.1), (7.2)."""
    instrument = instrument_arcsec or 0.0
    intrinsic = intrinsic_arcsec or 0.0
    K = true_width(fwhm_arcsec, instrument, intrinsic)
    phi = math.sqrt(instrument ** 2 + intrinsic ** 2)
    est = XrdEstimate(fwhm=fwhm_arcsec, instrument=instrument, intrinsic=intrinsic, K=K,
                      density=density_from_width(K, b_cm) if K == K else float("nan"),
                      upper_bound=phi == 0, limit=detection_limit(phi, accuracy_arcsec, b_cm) if phi else 0.0)
    if est.upper_bound:
        est.notes.append(f"Поправки на ширину прибора и естественную ширину не заданы: N_D = "
                         f"{est.density:.3g} см⁻² — только верхняя граница. Для Ge (111) с Cu Kα₁ "
                         f"естественная ширина {DARWIN_GE_111_ARCSEC:g}″ [KKA56]; для другого отражения и "
                         "прибора её нужно знать (кривая качания эталонного бездислокационного кристалла).")
    elif K != K:
        est.notes.append(f"Поправки ({phi:.3g}″) не меньше измеренной ширины ({fwhm_arcsec:g}″): уширения "
                         f"от дислокаций не видно — N_D ниже предела чувствительности ≈ {est.limit:.2g} см⁻².")
    elif est.density < est.limit:
        est.notes.append(f"N_D = {est.density:.3g} см⁻² ниже предела чувствительности (7.3) ≈ "
                         f"{est.limit:.2g} см⁻²: уширение сравнимо с точностью отсчёта, оценка ненадёжна.")
    return est


# ------------------------------------------------------------ образцы
@dataclass
class DefectSample:
    name: str
    epd: float | None = None          # ямки травления, см⁻²
    afm_density: float | None = None  # поверхностные дефекты по АСМ, см⁻²
    afm_rms: float | None = None      # шероховатость RMS, нм
    xrd_fwhm: float | None = None     # Δω₁/₂, угл. с
    xrd_instrument: float = 0.0       # φ прибора, угл. с
    xrd_intrinsic: float = 0.0        # естественная ширина, угл. с

    @property
    def xrd(self):
        if not self.xrd_fwhm:
            return None
        return xrd_density(self.xrd_fwhm, self.xrd_instrument, self.xrd_intrinsic)

    def densities(self):
        """Плотности по методам, см⁻² (нет данных — метод не входит)."""
        out = {}
        if self.epd:
            out[METHOD_EPD] = self.epd
        xrd = self.xrd
        if xrd is not None and xrd.density == xrd.density:
            out[METHOD_XRD] = xrd.density
        if self.afm_density:
            out[METHOD_AFM] = self.afm_density
        return out


def lifetime_by_method(sample, sigma_R):
    """τ_dis = 1/(σ_R·N) по (5.9), (5.10) [KKA56, ур. (3)] для плотности каждого метода, с."""
    return {m: ph.dislocation_lifetime(sigma_R, n) for m, n in sample.densities().items()}


def compare(sample):
    """Сопоставление методов для одного образца: отношения к EPD и пояснения."""
    d = sample.densities()
    notes = []
    epd, xrd_n, afm = d.get(METHOD_EPD), d.get(METHOD_XRD), d.get(METHOD_AFM)
    xrd = sample.xrd
    if xrd is not None:
        notes += xrd.notes
    if epd and xrd_n:
        r = epd / xrd_n
        text = f"EPD/N_XRD = {r:.3g}."
        if EPD_TO_XRD_111 * 0.8 <= r <= 1.25:
            text += (f" В пределах [KKA56]: от 1 до ≈ {EPD_TO_XRD_111:.2f} — травление не выявляет часть "
                     "наклонных и винтовых дислокаций, которые уширяют кривую качания.")
        elif r < EPD_TO_XRD_111 * 0.8:
            text += (f" Меньше ≈ {EPD_TO_XRD_111:.2f} [KKA56]: XRD завышает плотность — уширение не только "
                     "от дислокаций (не вычтены ширина прибора и естественная ширина, вклад подложки под "
                     "тонким слоем, изгиб пластины, мозаичность) — или травление пропускает дислокации.")
            if xrd.upper_bound or xrd.density < 3 * xrd.limit:
                text += (" Здесь XRD-оценка близка к пределу чувствительности или без поправок, поэтому "
                         "расхождение объясняется в первую очередь уширением не от дислокаций.")
        else:
            text += (" Больше 1: ямки дают не только дислокации (преципитаты, включения, загрязнения, "
                     "повреждения поверхности) или XRD-поправки завышены.")
        notes.append(text)
    if epd and afm:
        r = afm / epd
        notes.append(f"N_АСМ/EPD = {r:.3g}. АСМ считает все особенности поверхности (выходы дислокаций, "
                     "ямки и холмики роста, частицы) без избирательного травления; связи с дислокациями "
                     "в источниках проекта нет — только эмпирическая калибровка по образцам.")
    if epd and not (KKA_RANGE[0] <= epd <= KKA_RANGE[1]):
        notes.append(f"EPD = {epd:.3g} см⁻² вне диапазона {KKA_RANGE[0]:.0e}–{KKA_RANGE[1]:.0e} см⁻², где "
                     "[KKA56] сверяли травление с XRD: соответствие методов здесь не проверено.")
    if xrd is not None and epd and xrd.upper_bound:
        limit = detection_limit(DARWIN_GE_111_ARCSEC)
        if epd < limit:
            notes.append(f"Для сравнения: при поправке порядка естественной ширины Ge (111) "
                         f"{DARWIN_GE_111_ARCSEC:g}″ предел чувствительности XRD (7.3) ≈ {limit:.2g} см⁻², "
                         f"а EPD = {epd:.3g} см⁻² ниже него. Вероятно, кривая качания этого образца почти "
                         "целиком — ширина прибора и отражения, а не дислокации; задайте поправки (п. 14.2).")
    if xrd is not None and epd and xrd.limit and epd < xrd.limit:
        notes.append(f"EPD = {epd:.3g} см⁻² ниже предела чувствительности XRD ≈ {xrd.limit:.2g} см⁻² (7.3): "
                     "такую плотность кривая качания не различает — калибровка XRD по этому образцу "
                     "невозможна, нужны образцы с большей плотностью дислокаций.")
    return {"densities": d, "notes": notes}


# ------------------------------------------------------------ калибровка
@dataclass
class Calibration:
    x_method: str
    n: int                    # число образцов
    slope: float              # k в log10(EPD) = log10(c) + k·log10(X)
    intercept: float          # log10(c)
    slope_err: float
    r2: float
    ratio: float              # среднее геометрическое EPD/X (одноточечная калибровка)
    samples: list

    def predict(self, x):
        """EPD по неразрушающему значению x (та же величина, что при калибровке)."""
        if self.n >= 2 and np.isfinite(self.slope):
            return 10 ** (self.intercept + self.slope * math.log10(x))
        return self.ratio * x


X_VALUES = {
    "xrd_density": ("N_XRD по (7.2), см⁻²", lambda s: s.densities().get(METHOD_XRD)),
    "afm_density": ("плотность дефектов АСМ, см⁻²", lambda s: s.afm_density),
    "afm_rms": ("RMS АСМ, нм", lambda s: s.afm_rms),
    "xrd_fwhm": ("Δω₁/₂ измеренная, угл. с", lambda s: s.xrd_fwhm),
}


def usable_for_calibration(sample, x_key):
    """Для калибровки по N_XRD годятся только оценки с поправками (7.1) и выше
    предела чувствительности (7.3): верхняя граница калибровку искажает."""
    if x_key != "xrd_density":
        return True
    xrd = sample.xrd
    return xrd is not None and not xrd.upper_bound and xrd.density == xrd.density \
        and xrd.density >= xrd.limit


def calibrate(samples, x_key="xrd_density"):
    """Калибровка EPD по неразрушающему методу: степенная зависимость EPD = c·X^k
    (наименьшие квадраты в логарифмах) по образцам, где есть обе величины.
    Один образец — только множитель c = EPD/X (k = 1 по [KKA56] для XRD).
    Для N_XRD берутся только оценки с поправками (usable_for_calibration)."""
    _label, getter = X_VALUES[x_key]
    samples = [s for s in samples if usable_for_calibration(s, x_key)]
    pairs = [(s, getter(s), s.epd) for s in samples]
    pairs = [(s, x, y) for s, x, y in pairs if x and y and x > 0 and y > 0]
    if not pairs:
        return None
    lx = np.log10([x for _s, x, _y in pairs])
    ly = np.log10([y for _s, _x, y in pairs])
    ratio = float(10 ** np.mean(ly - lx))
    slope = intercept = slope_err = r2 = float("nan")
    if len(pairs) >= 2 and np.ptp(lx) > 0:
        slope, intercept = np.polyfit(lx, ly, 1)
        fit = intercept + slope * lx
        ss_res = float(np.sum((ly - fit) ** 2))
        ss_tot = float(np.sum((ly - ly.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        if len(pairs) >= 3:
            slope_err = math.sqrt(ss_res / (len(pairs) - 2) / float(np.sum((lx - lx.mean()) ** 2)))
    return Calibration(x_method=x_key, n=len(pairs), slope=float(slope), intercept=float(intercept),
                       slope_err=slope_err, r2=r2, ratio=ratio, samples=[s.name for s, _x, _y in pairs])


def sample_from_params(name, params):
    """Образец из параметров набора (ключи presets): N_dis — ямки травления."""
    def value(key):
        v = params.get(key)
        return float(v) if v not in (None, "") else None
    return DefectSample(name=name, epd=value("N_dis") or None, afm_density=value("afm_defects"),
                        afm_rms=value("afm_rms_nm"), xrd_fwhm=value("xrd_fwhm"),
                        xrd_instrument=value("xrd_instrument") or 0.0,
                        xrd_intrinsic=value("xrd_intrinsic") or 0.0)


# Заголовки таблицы образцов (CSV/Excel) → поле DefectSample.
TABLE_COLUMNS = {
    "образец": "name", "sample": "name", "name": "name",
    "epd": "epd", "ямки": "epd", "травление": "epd",
    "afm": "afm_density", "асм": "afm_density", "афм": "afm_density",
    "rms": "afm_rms",
    "fwhm": "xrd_fwhm", "xrd": "xrd_fwhm", "δω": "xrd_fwhm", "dw": "xrd_fwhm",
    "прибор": "xrd_instrument", "instrument": "xrd_instrument", "φ": "xrd_instrument",
    "естеств": "xrd_intrinsic", "intrinsic": "xrd_intrinsic", "darwin": "xrd_intrinsic",
}


def samples_from_table(path):
    """Образцы из таблицы (CSV/TXT/Excel): первая строка — заголовок; столбцы
    «образец», «EPD», «АСМ», «RMS», «FWHM» (угл. с), «прибор», «естеств.»."""
    from mesa_diode.simulator import datafile

    report = datafile.DataReport(path=path, kind="defects")
    rows = [r for r in datafile.read_table(path, report) if any(c not in (None, "") for c in r)
            and not str(r[0] or "").lstrip().startswith("#")]          # комментарии пропускаются
    if report.errors or not rows:
        raise ValueError(report.text(("error",)) or "таблица пустая")
    header = [str(c or "").strip().lower() for c in rows[0]]
    columns = {}
    for index, name in enumerate(header):
        for prefix, attr in TABLE_COLUMNS.items():
            if name.startswith(prefix) and attr not in columns:
                columns[attr] = index
                break
    if "epd" not in columns and "xrd_fwhm" not in columns:
        raise ValueError("В заголовке нет столбцов EPD или FWHM. Подпишите столбцы: образец, EPD, АСМ, "
                         "RMS, FWHM (угл. с), прибор, естеств. Подробнее: методичка, п. 14.5.")
    samples = []
    for row in rows[1:]:
        values = {}
        for attr, index in columns.items():
            cell = row[index] if index < len(row) else None
            values[attr] = (str(cell).strip() if cell is not None else "") if attr == "name" \
                else datafile.parse_number(cell)
        name = values.pop("name", "") or f"образец {len(samples) + 1}"
        samples.append(DefectSample(name=name, **{k: v for k, v in values.items() if v is not None}))
    return samples
