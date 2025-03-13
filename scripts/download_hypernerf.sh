#!/bin/bash

# Desc: Download the HyperNeRF models from the official release.
# Args: $1: Destination path to save the models. e.g. data/hypernerf

dest_path=$1
mkdir -p "${dest_path}"

# download-list
files=(
  "interp_chickchicken.zip"
  "interp_torchocolate.zip"
  "misc_americano.zip"
  "misc_espresso.zip"
  "misc_keyboard.zip"
  "misc_split-cookie.zip"
)

# download
for file in "${files[@]}"; do
    url="https://github.com/google/hypernerf/releases/download/v0.1/${file}"
    zip_path="${dest_path}/${file}"

    # check exist
    if [ -f "${zip_path}" ]; then
        echo "File ${file} already exists. Skipping download."
    else
        echo "Downloading ${file}..."
        wget -q --show-progress "${url}" -P "${dest_path}"
        if [ $? -ne 0 ]; then
            echo "Failed to download ${file}. Skipping..."
            continue
        fi
    fi

    # unzip
    echo "Extracting ${file}..."
    unzip -o "${zip_path}" -d "${dest_path}"
    if [ $? -eq 0 ]; then
        rm "${zip_path}"
        echo "Extraction successful. Deleted ${file}."
    else
        echo "Extraction failed for ${file}. Keeping the zip file for debugging."
    fi
done

echo "Done."


