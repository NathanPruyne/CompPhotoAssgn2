#!/bin/bash

INPUT_DIR="../data/noise/dark"
OUTPUT_DIR="../data/noise/dark"

for img in "$INPUT_DIR"/*.nef; do
    
    # Extract the filename without the path and extension
    base_name=$(basename "$img" | sed 's/\.[^.]*$//')
    
    echo "dcraw on $base_name"
    
    dcraw -4 -T -g 1 1 -w -q 3 -o 1 -c "$img" > "$OUTPUT_DIR/${base_name}.tiff"
done

rename 's/\d+/sprintf("%02d",$&)/e' "$OUTPUT_DIR"/*