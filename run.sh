#!/bin/bash
set -eo pipefail

KEY_PATH=$(mktemp -t git-crypt-key-XXXXXX)

if [ -n "$GIT_KEY" ]; then
    echo "$GIT_KEY" > "$KEY_PATH"
elif [ -f "$GIT_KEY_FILE" ]; then
    cp "$GIT_KEY_FILE" "$KEY_PATH"
else
    echo "No GIT_KEY or GIT_KEY_FILE found. Skipping unlock."
fi

if [ -f "$KEY_PATH" ]; then
    if [ ! -d ".git" ]; then
        git init -q
        echo "*" > .git/info/exclude
    fi

    git-crypt unlock "$KEY_PATH"
    rm -f "$KEY_PATH"
fi

exec "$@"
