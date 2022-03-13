# !/bin/bash

CONFIG_FILE_PATH="$(git rev-parse --show-toplevel)/mypy.ini"
echo ${CONFIG_FILE_PATH}
CONFIG_FILE_OPTION="--config-file=${CONFIG_FILE_PATH}"

find . -not -path "*/.*/*" \
    -not -path "setup.py" \
    -not -path "__init__.py" \
    -regextype sed -regex ".*/.*[^_].py" \
    | xargs python3 -m mypy ${CONFIG_FILE_OPTION} $* \
    | grep "^[^/SF]" | uniq