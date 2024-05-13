#!/bin/bash

# Source directory where the folders currently reside
src_dir="/gpfs/data/ranganathlab/sym_10"

# Destination directory where the folders should be moved
dest_dir="/gpfs/scratch/as16583/ckpts/high_dim"

# Loop through each item in the source directory
for src_folder in "$src_dir"/*; do
    # Extract the folder name
    folder_name=$(basename "$src_folder")

    # Check if the destination already contains this folder
    if [ -d "$dest_dir/$folder_name" ]; then
        echo "Directory already exists, not moving: $folder_name"
    else
        # Move the directory if it does not exist in the destination
        mv "$src_folder" "$dest_dir"
        echo "Moved directory: $folder_name"
    fi
done

echo "Operation completed. All applicable directories have been moved from $src_dir to $dest_dir."