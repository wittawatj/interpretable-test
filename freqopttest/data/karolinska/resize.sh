#!/bin/bash 

size="48"
Src="S_crop"
Dest="crop_${size}"
for f in $(find $Src -iname "*.jpg"); 
do 
    echo "processing $f"
    dest_file=$Dest/$(basename $f)
    convert -resize ${size}x${size} $f  $dest_file; 
    # change to grayscale 
    mogrify -colorspace gray $dest_file

done
