# mesa-diode-app

Программа моделирования вольт-амперных характеристик мезадиодных структур
n⁺/i/p на основе германия и твёрдых растворов Ge₁₋ₓSnₓ.

Задача — связать воедино результаты разных методов диагностики эпитаксиальных
слоёв (ВАХ, C–V-профилирование, эффект Холла, рентгеновская дифракция, АСМ)
в единое параметрическое описание образца, по которому можно предсказывать
характеристики новых ростов без разрушающих методов.

## Данные

**Экспериментальные данные не входят в этот репозиторий.** Они хранятся
отдельно и подключаются по пути из переменной окружения `MESA_DATA_DIR`.
В коде нет ни числовых параметров конкретных образцов, ни путей к ним:
всё читается из паспорта образца `samples/<id>/meta.yaml` в каталоге данных.

## Установка

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env           # Windows: copy .env.example .env
# затем впишите в .env путь к каталогу с данными
```

## Использование

```python
from mesa_diode.loaders import load_meta, load_iv
from mesa_diode.diode import ideality_factor, mesa_area_cm2
from mesa_diode.fit import fit_two_diode

meta = load_meta("<sample-id>")
voltage, current = load_iv("<sample-id>")

print("Коэффициент идеальности:", ideality_factor(voltage, current))
print("Площадь мезы, см²:", mesa_area_cm2(meta))

result = fit_two_diode(voltage, current, with_shunt=True)
print("Диффузионный ток насыщения, А:", result.diffusion_saturation_A)
print("Рекомбинационный ток насыщения, А:", result.recombination_saturation_A)
```

## Физическая модель

Двухдиодная модель раздельно описывает два механизма протекания тока:

```
I(U) = I₀₁ · [exp(qU / kT) − 1] + I₀₂ · [exp(qU / 2kT) − 1] + U / R_sh
```

- **I₀₁, n = 1** — диффузионный ток, идеальный случай
- **I₀₂, n = 2** — рекомбинация в области пространственного заряда через
  дефектные центры; рост этой компоненты прямо указывает на структурные дефекты
- **R_sh** — шунтирующее сопротивление (утечки по периметру мезы)

Эффективный коэффициент идеальности между 1 и 2 показывает, какой механизм
преобладает. Извлекается по наклону `ln(I)` от `U` на участке `U ≫ n·kT/q`.

Последовательное сопротивление пока не учитывается — при его включении
уравнение становится неявным относительно тока.

## Структура

| Модуль | Назначение |
|---|---|
| `mesa_diode/config.py` | Граница «код ↔ данные»: разрешение `MESA_DATA_DIR` |
| `mesa_diode/loaders.py` | Чтение паспортов образцов и файлов измерений |
| `mesa_diode/diode.py` | Уравнение Шокли, двухдиодная модель, геометрия мезы |
| `mesa_diode/fit.py` | Подгонка параметров модели под измеренную ВАХ |
| `mesa_diode/simulator/defects.py` | Сопоставление методов контроля дислокаций (травление, АСМ, XRD) и калибровка XRD по травлению [KKA56]; окно — `defects_window.py` |
| `mesa_diode/simulator/models.py` | Реестр моделей тока симулятора: эмпирическая, двухдиодная, физическая; подключение своей модели |
| `mesa_diode/simulator/` | GUI-симулятор мезадиода Ge — прямой расчёт ВАХ, ВФХ и J–V по параметрам структуры. См. `mesa_diode/simulator/README.md` |
| `tests/` | Проверки на синтетических данных |

## GUI-симулятор мезадиода Ge (ВАХ, ВФХ, J–V)

Отдельное интерактивное приложение: задаёте параметры мезы — сразу видите
расчётные ВАХ и ВФХ, с возможностью наложить экспериментальные точки.
Не требует `MESA_DATA_DIR` — это прямая модель, а не подгонка под образец. При
старте открывается опорный образец — модельная структура (не реальный образец);
если каталог данных задан в `.env`, из него открываются наборы образцов, а
методичка — из меню «Справка».

```bash
python scripts/run_simulator.py
```

Окно устроено по шагам работы: слева — режим, образец, параметры и кнопки
«Рассчитать», **«Подогнать к ВАХ»**, «Заполнить пустые поля»; справа —
графики и вкладки результатов. Режимы: **Базовая модель** — ВАХ по
эмпирической формуле с n и J₀; **Расширенная модель** — физическая модель по
всему, что измеряется; **Подгонка** — плюс неизмеряемые параметры формул.
Автоподгонка находит неизмеряемые параметры по загруженной ВАХ, сама решает,
нужны ли нелинейная утечка и модуляция последовательного сопротивления, и
объясняет выбор; график отклонений показывает, где модель расходится с
экспериментом. Экспериментальные ВАХ и ВФХ загружаются из CSV, TXT или
Excel; программа проверяет файл и подсказывает, что не так. Формат и
примеры — [`mesa_diode/simulator/examples/README.md`](mesa_diode/simulator/examples/README.md).
Что изменилось по версиям — [`CHANGELOG.md`](CHANGELOG.md).

Подробности запуска и сборки в самостоятельный `.exe` — в
[`mesa_diode/simulator/README.md`](mesa_diode/simulator/README.md).

## Быстрый запуск с авто-обновлением (Windows)

Если код (`mesa-diode-app`) и данные (каталог из `MESA_DATA_DIR`) лежат
локально как git-репозитории, `launch.bat` перед каждым запуском сам
подтягивает актуальную версию обоих и запускает симулятор:

```
launch.bat
```

(двойной клик в проводнике или запуск из PowerShell). Делает по порядку:
обновляет оба репозитория (`git pull`, безопасно — не трогает репозиторий,
если там есть несохранённые изменения или история разошлась), создаёт
`.venv` при первом запуске, ставит/обновляет зависимости, запускает
симулятор. Требует уже настроенный `.env` (см. «Установка» выше).

Логика — в `scripts/dev_launch.py`, можно запустить и напрямую:
```bash
python scripts/dev_launch.py
```

## Тесты

```bash
pytest
```

Тесты работают только на синтетических данных, посчитанных самой моделью
с известными параметрами, и не требуют доступа к `MESA_DATA_DIR`. Тесты,
которым нужны данные (эталоны §9 на наборе реального образца, ссылки подсказок
на пункты методички), без `MESA_DATA_DIR` пропускаются.

## Литература

1. Shockley W. The Theory of p–n Junctions in Semiconductors and p–n Junction
   Transistors // Bell System Technical Journal. 1949. Vol. 28. P. 435–489.
2. Sah C.T., Noyce R.N., Shockley W. Carrier Generation and Recombination in
   p–n Junctions and p–n Junction Characteristics // Proceedings of the IRE.
   1957. Vol. 45. P. 1228–1243. DOI: 10.1109/JRPROC.1957.278528
3. Giunto A., Fontcuberta i Morral A. Defects in Ge and GeSn and their impact
   on optoelectronic properties // Applied Physics Reviews. 2024. Vol. 11.
   Art. 041333. DOI: 10.1063/5.0218623
4. Dolabella S., Borzì A., Dommann A., Neels A. Lattice Strain and Defects
   Analysis in Nanostructured Semiconductor Materials and Devices by
   High-Resolution X-Ray Diffraction // Small Methods. 2022. Vol. 6.
   Art. 2100932. DOI: 10.1002/smtd.202100932
5. Miao Y.-H., Hu H.-Y., Li X. et al. Evaluation of threading dislocation
   density of strained Ge epitaxial layer by high resolution x-ray
   diffraction // Chinese Physics B. 2017. Vol. 26, № 12. Art. 127309.
   DOI: 10.1088/1674-1056/26/12/127309
