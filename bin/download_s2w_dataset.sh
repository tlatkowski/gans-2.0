#!/usr/bin/env bash
S2W_DATA_LINK=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/summer2winter_yosemite.zip

S2W_FILE=summer2winter_yosemite.zip

DATA_DIR=data
mkdir -p ${DATA_DIR}

wget --no-check-certificate ${S2W_DATA_LINK} -O ${DATA_DIR}/${S2W_FILE}
unzip ${DATA_DIR}/${S2W_FILE} -d ${DATA_DIR}