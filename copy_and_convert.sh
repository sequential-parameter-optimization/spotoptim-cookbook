#!/bin/bash

# Source and destination directories
SOURCE_DIR="/Users/bartz/workspace/spotoptim/docs/manuals"
DEST_DIR="/Users/bartz/workspace/spotoptim-cookbook"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

# Check if destination directory exists
if [ ! -d "$DEST_DIR" ]; then
    echo "Error: Destination directory $DEST_DIR does not exist"
    exit 1
fi

# Copy all files from source to destination
echo "Copying files from $SOURCE_DIR to $DEST_DIR..."

# Use find to get all files (not directories) in the source directory
find "$SOURCE_DIR" -type f | while read -r file; do
    # Get the relative path from the source directory
    rel_path="${file#$SOURCE_DIR/}"
    
    # Get the directory part and filename
    dir_part=$(dirname "$rel_path")
    filename=$(basename "$rel_path")
    
    # Create destination directory if it doesn't exist
    if [ "$dir_part" != "." ]; then
        mkdir -p "$DEST_DIR/$dir_part"
    fi
    
    # Check if file has .md extension
    if [[ "$filename" == *.md ]]; then
        # Change .md to .qmd
        new_filename="${filename%.md}.qmd"
        dest_path="$DEST_DIR/$dir_part/$new_filename"
        echo "  $rel_path -> $dir_part/$new_filename"
    else
        # Copy with original name
        dest_path="$DEST_DIR/$rel_path"
        echo "  $rel_path"
    fi
    
    # Copy the file
    cp "$file" "$dest_path"
done

echo "Done! All files copied successfully."
