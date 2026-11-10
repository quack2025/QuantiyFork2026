"""Banner detection by whole words (2026-11-10).

`PATTERNS` only matched a word joined by `-`/`_`, so a label with spaces
("¿En cuál ciudad vive?") never matched and many files got no banner at all.
These tests pin the extra pass in `_detect_banners`: what it finds, what it
must not find, and that it never changes a question's type.
"""
import pandas as pd
import pytest

from quantipymrx.core.dataset import DataSet
from quantipymrx.core.metadata import ValueLabel, VariableType
from quantipymrx.analysis.auto_detect import PatternDetector, suggest_banners, BannerType

TK = "es-ES"


def make(columns):
    """columns: {name: (label, [value labels] or None, data list)}"""
    df = pd.DataFrame({name: data for name, (_, _, data) in columns.items()})
    ds = DataSet.from_dataframe(df, name="t", text_key=TK)
    for name, (label, values, _) in columns.items():
        meta = ds.meta.columns[name]
        meta.text = {TK: label}
        if values is not None:
            meta.type = VariableType.SINGLE
            meta.values = [ValueLabel(value=i + 1, text={TK: v}) for i, v in enumerate(values)]
    return ds


def banner_map(ds):
    return {b.variable: b.banner_type for b in suggest_banners(ds, max_banners=20)}


N = 40
two = [1, 2] * (N // 2)
four = [1, 2, 3, 4] * (N // 4)


def test_profile_questions_with_spaces_and_accents():
    ds = make({
        "F1": ("¿En qué ciudad vive usted?", ["Ciudad A", "Ciudad B"], two),
        "F2": ("Es usted…?", ["Hombre", "Mujer"], two),
        "F3R": ("P3 recodificada", ["18 a 24", "25 a 34", "35 a 44", "45 a 55"], four),
        "F4": ("Estrato socioeconómico de la vivienda", ["1-2", "3-4", "5-6", "NS"], four),
        "Q1": ("¿Qué tanto le agrada el producto?", ["Nada", "Poco", "Algo", "Mucho"], four),
    })
    got = banner_map(ds)
    assert got["F1"] == BannerType.REGION
    assert got["F2"] == BannerType.GENDER
    assert got["F3R"] == BannerType.AGE_GROUP
    assert got["F4"] == BannerType.INCOME
    assert "Q1" not in got


def test_names_that_already_worked_still_do():
    ds = make({
        "SEXO": ("Sexo", ["Hombre", "Mujer"], two),
        "edad_grupo": ("Grupo", ["18-24", "25-34", "35-44", "45+"], four),
        "GENDER": ("Gender", ["Male", "Female"], two),
    })
    got = banner_map(ds)
    assert got["SEXO"] == BannerType.GENDER
    assert got["GENDER"] == BannerType.GENDER
    assert "edad_grupo" in got


@pytest.mark.parametrize("label,values", [
    ("¿Qué género musical prefiere?", ["Rock", "Salsa"]),
    ("¿Es usted cliente del banco?", ["Sí", "No"]),
    ("¿Qué edad tiene su hijo menor?", ["0 a 5", "6 a 11", "12 a 17", "18 a 25"]),
    ("¿Cuánto pagaría al mes?", ["$10.000 - $20.000", "$20.000 - $30.000", "$30.000 - $40.000", "Más de $40.000"]),
    ("¿Con qué frecuencia compra?", ["1-2 veces", "3-5 veces", "6-10 veces", "Más"]),
    ("Calidad del servicio en sucursal", ["Mala", "Regular", "Buena", "Excelente"]),
    ("Opción B:¿Cuáles de estas situaciones vivió?", ["No seleccionado", "Opción B"]),
    ("Ciudad:¿En cuáles ciudades ha vivido?", ["0", "1"]),
    ("Pensando en su seguro contra la pérdida de ingresos, ¿qué tan satisfecho está?", ["Nada", "Poco", "Algo", "Mucho"]),
])
def test_not_profile(label, values):
    data = (list(range(1, len(values) + 1)) * N)[:N]
    ds = make({"P9": (label, values, data)})
    assert banner_map(ds) == {}


def test_income_as_a_phrase():
    ds = make({"P20": ("¿Cuál es el ingreso mensual de su hogar?", ["Bajo", "Medio", "Alto"], ([1, 2, 3] * N)[:N])})
    assert banner_map(ds) == {"P20": BannerType.INCOME}


def test_not_profile_series_item():
    ds = make({"P9_6": ("Producto F: Nivel de ingresos", ["Nada", "Poco", "Algo", "Mucho"], four)})
    assert banner_map(ds) == {}


def test_a_second_question_of_a_kind_already_found_is_not_added():
    ds = make({
        "SEXO": ("Sexo", ["Hombre", "Mujer"], two),
        "Q15": ("¿Con qué género se identifica usted?", ["Hombre", "Mujer", "Otro", "Prefiero no decir"], four),
        "INGRESO_NIVEL": ("Nivel de ingresos", ["Bajo", "Medio", "Alto", "NS"], four),
    })
    got = banner_map(ds)
    assert got["SEXO"] == BannerType.GENDER
    assert "Q15" not in got
    assert got["INGRESO_NIVEL"] == BannerType.INCOME


def test_numeric_age_is_never_a_banner():
    ages = list(range(18, 18 + N))
    ds = make({"EDAD": ("Edad", None, ages), "edad_r": ("Edad en rangos", ["18-24", "25-34", "35-44", "45+"], four)})
    # The numeric one has a single label for the non-answer, like many files.
    ds.meta.columns["EDAD"].values = [ValueLabel(value=99, text={TK: "No responde"})]
    got = banner_map(ds)
    assert "EDAD" not in got
    assert "edad_r" in got


def test_question_types_do_not_change():
    ds = make({
        "P10": ("¿Recomendaría la marca en su ciudad?", [str(i) for i in range(11)], (list(range(1, 12)) * N)[:N]),
        "F2": ("Es usted…?", ["Hombre", "Mujer"], two),
    })
    det = PatternDetector(ds)
    det._detect_individual_variables()
    before = {k: v.question_type for k, v in det._detected_variables.items()}
    det._detect_banners()
    assert {k: v.question_type for k, v in det._detected_variables.items()} == before
