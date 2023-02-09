#!/usr/bin/env bash

# Script settings
UPLOADER_DIR=/space/drwho/3/neurobooth/applications/opt/Dropbox-Uploader
LOCAL_SRC=/space/neo/3/neurobooth/reports/__SUMMARY__
REMOTE_DST=PROM_Reports
CONDA_ENV=reports
CONDA_DIR=/space/drwho/3/neurobooth/applications/opt/miniconda3
FILES=(prom_completion.csv prom_contact.csv prom_completion_time.png prom_completion_time_stats.csv)

source $CONDA_DIR/etc/profile.d/conda.sh
conda activate $CONDA_ENV
TIME="$(date +%Y-%m-%d_%Hh-%Mm-%Ss)"
prom_completion_report
conda deactivate

cd $LOCAL_SRC || exit 1
for file in "${FILES[@]}"
do
    basename=${file%.*}    # Remove extension
    extension=${file##*.}  # Remove basename
    new_file="$basename"_"$TIME.$extension"
    cp "$file" "$new_file"
    $UPLOADER_DIR/dropbox_uploader.sh upload "$new_file" $REMOTE_DST
    rm "$new_file"
done
