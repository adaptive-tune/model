#/bin/sh
set -eo pipefail

base_url="https://os.unil.cloud.switch.ch/fma"
metadata_file="fma_metadata.zip"
dataset_file="fma_small.zip"

metadata_sum="f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
dataset_sum="ade154f733639d52e35e32f5593efe5be76c6d70"

#wget "${base_url}/${metadata_file}"
#wget "${base_url}/${dataset_file}"

echo "${metadata_sum} ${metadata_file}" | sha1sum --check
echo "${dataset_sum} ${dataset_file}" | sha1sum --check

unzip ${metadata_file}
unzip ${dataset_file}
