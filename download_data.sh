#!/bin/bash
dest_dir=$1
if [ -z $dest_dir ]; then
  echo "Error! No destination directory."
  exit 125
fi
package_names="gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip"
python $PWD/third-party/cityscapesScripts/cityscapesscripts/download/download.py -d $dest_dir $package_names