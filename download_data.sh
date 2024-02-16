#!/bin/bash
current_dir="$(cd $(dirname $0) && pwd)"
third_party_dir="$current_dir/third-party"
# check if third-party directory needs to be created
if [ ! -d "$third_party_dir" ]; then
  echo "Creating $third_party_dir"
  mkdir $third_party_dir
fi
# check if cityscapesScripts repo needs to be cloned
cityscapesScripts_dir="$third_party_dir/cityscapesScripts"
if [ ! -d "$cityscapesScripts_dir" ]; then
  echo "cloning cityscapesScripts repository into third-party"
  # forked version of the current master of the cityscapesScripts repository (https://github.com/mcordts/cityscapesScripts/tree/master?tab=readme-ov-file)
  git clone https://github.com/ChrisKelton/cityscapesScripts--02152024.git $cityscapesScripts_dir
fi
package_names="gtFine_trainvaltest.zip leftImg8bit_trainvaltest.zip"
dest_dir="$current_dir/data"
download_data=false
stamp_file="$dest_dir/data.stamp"
unzip_files=false
# check if stamp file exists (claims that data has been unzipped properly)
if [ ! -f "$stamp_file" ]; then
  unzip_files=true
  # check to see if data needs to be downloaded
  if [ ! -d "$dest_dir" ]; then
    echo "Creating $dest_dir"
    mkdir $dest_dir
    download_data=true
  else
    # check for each required zip file (if any are not found, then invoke downloader.py script)
    for package_name in $package_names; do
      zip_file="$dest_dir/$package_name"
      echo "Checking if $zip_file exists."
      if [ ! -f "$zip_file" ]; then
        download_data=true
      fi
    done
  fi
  if $download_data; then
    echo "Downloading CityScapes DataSet for Semantic Segmentation"
    python $cityscapesScripts_dir/cityscapesscripts/download/downloader.py -d $dest_dir $package_names
  fi
else
  # verify stamp file says files were successfully unzipped
  echo "Checking stamp file at '${stamp_file}'"
  success="$(cat $stamp_file | grep "Successfully unzipped the data" | wc -l)"
  if [ $success -lt 1 ]; then
    unzip_files=true
  fi
fi
#
if $unzip_files; then
  echo "Unzipping CityScapes DataSet..."
  python $current_dir/data_prep/cityscape_dataset.py
fi
#
echo "Finished"
