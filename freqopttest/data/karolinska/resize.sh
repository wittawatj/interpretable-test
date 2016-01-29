#!/bin/bash 

size="48"
Dest="S_${size}"
for f in $(find KDEF_straight -iname "*.jpg"); 
do 
    echo "processing $f"
    dest_file=$Dest/$(basename $f)
    convert -resize ${size}x${size} $f  $dest_file; 
    # change to grayscale 
    mogrify -colorspace gray $dest_file

done
