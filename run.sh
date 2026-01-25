#!/bin/bash
set -eo pipefail

KEY_PATH=$(mktemp -t git-crypt-key-XXXXXX)

if [ -n "$GIT_KEY" ]; then
    echo "$GIT_KEY" > "$KEY_PATH"
elif [ -f "$GIT_KEY_FILE" ]; then
    cp "$GIT_KEY_FILE" "$KEY_PATH"
else
    echo "No GIT_KEY or GIT_KEY_FILE found. Skipping unlock."
    rm -f "$KEY_PATH"
    KEY_PATH=""
fi

if [ -f "$KEY_PATH" ]; then
    if [ ! -d ".git" ]; then
        git init -q
    fi

    git ls-files --cached --others --exclude-standard | git check-attr --stdin filter | while read -r f; do
      if grep -q 'filter: git-crypt' <<< "$f" ; then
        file=$(cut -d: -f1 <<< "$f")
        if [ -f "$file" ]; then
          echo "decrypting $file"
          cat "$file" | git-crypt smudge --key-file=$KEY_PATH > "$file".dec
          mv "$file".dec "$file"
        fi
      fi
    done

    rm -f "$KEY_PATH"
fi

exec "$@"
