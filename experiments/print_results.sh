#!/bin/bash

# Change this to your target parent directory, or use "$1" to accept as argument
PARENT_DIR="${1:-.}"

# Find all files named "settings" in subdirectories of the parent directory
find "$PARENT_DIR" -type f -name "settings.txt" | while read -r settings_file; do
  echo "==== Contents of: $settings_file ===="
  cat "$settings_file"
  echo # newline for separation
done

