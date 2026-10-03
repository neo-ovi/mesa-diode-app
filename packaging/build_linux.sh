#!/usr/bin/env bash
# Сборка программы для Linux, два варианта:
#   dist/MesaSimulator                      — один файл (при запуске распаковывается
#                                             во временную папку);
#   dist/MesaSimulator-linux-x86_64.tar.xz  — папка MesaSimulator/ с исполняемым
#                                             файлом и инструкцией README-Linux.txt:
#                                             архив почти вдвое меньше, запуск быстрее,
#                                             признак «исполняемый» сохраняется.
#
# Собирается в Docker-контейнере Ubuntu 22.04. Программа, собранная на старой
# glibc (2.35), работает на новых системах — Linux Mint 21.x и 22.x, LMDE 6 и 7,
# Ubuntu 22.04 и новее, Debian 12 и новее; собранная на новой — на старых нет.
#
# Запуск из корня репозитория:   bash packaging/build_linux.sh
# Нужен Docker (sudo apt install docker.io; пользователь — в группе docker).
# DOCKER_ARGS — дополнительные ключи docker run (например, --network host).
# Без Docker — на самой Ubuntu 22.04 / Mint 21: см. mesa_diode/simulator/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE="${IMAGE:-ubuntu:22.04}"
mkdir -p dist
# shellcheck disable=SC2086  # DOCKER_ARGS — несколько ключей
docker run --rm ${DOCKER_ARGS:-} -v "$PWD:/src:ro" -v "$PWD/dist:/out" "$IMAGE" bash -euo pipefail -c '
    export DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
    apt-get update -qq
    apt-get install -y -qq --no-install-recommends \
        python3 python3-venv python3-dev python3-tk tk8.6 binutils xz-utils >/dev/null
    python3 -m venv /tmp/venv
    /tmp/venv/bin/pip install -q --upgrade pip
    /tmp/venv/bin/pip install -q -r /src/requirements-build.txt
    mkdir /tmp/src
    tar -C /src --exclude=./.git --exclude=./.venv --exclude=./dist --exclude=./build -cf - . | tar -C /tmp/src -xf -
    cd /tmp/src
    build() {   # $1 — каталог результата; MESA_ONEDIR задаёт вариант
        /tmp/venv/bin/pyinstaller --noconfirm --clean --log-level WARN \
            --distpath "$1" --workpath /tmp/build packaging/mesa_simulator.spec
    }
    build /tmp/onefile
    MESA_ONEDIR=1 build /tmp/onedir
    /tmp/onefile/MesaSimulator --version
    /tmp/onedir/MesaSimulator/MesaSimulator --version
    cp packaging/README-Linux.txt /tmp/onedir/MesaSimulator/
    tar -C /tmp/onedir -cf - MesaSimulator | xz -9e -T0 > /out/MesaSimulator-linux-x86_64.tar.xz
    cp /tmp/onefile/MesaSimulator /out/
    chown "$(stat -c %u:%g /out)" /out/MesaSimulator /out/MesaSimulator-linux-x86_64.tar.xz
'
ls -lh dist/MesaSimulator dist/MesaSimulator-linux-x86_64.tar.xz
