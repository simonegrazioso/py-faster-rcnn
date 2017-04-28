#!/bin/bash

BASE_DIR=$1

DIR_ANNOT=Annotations
IM_DIR=Images

NUM_TEST=30
MAX_TRAIN=250

SET_TRAIN=ImageSets/trainval.txt
SET_TEST=ImageSets/test.txt

cd $BASE_DIR
rm -f $SET_TRAIN $SET_TEST
for dir in `find $DIR_ANNOT -type d \( ! -name $DIR_ANNOT \) -exec basename {} \;`;
do
   SYN_DIR=./$DIR_ANNOT/$dir
   NUM_ANNOT=`find $SYN_DIR -iname "*.xml" | wc -l`
   NUM_TRAIN=$[$NUM_ANNOT-$NUM_TEST]

   for ff in `find $SYN_DIR -iname "*.xml" | head -n $(($NUM_TRAIN<$MAX_TRAIN?$NUM_TRAIN:$MAX_TRAIN))`;
   do
   fbase=`basename $ff .xml`
   name=$IM_DIR/$(echo $ff | cut -f 3,4 -d '/' | cut -f 1 -d '.').JPEG
   if [[ -a ${name} ]]; then
	   echo $fbase >> $SET_TRAIN
   fi
   done
   for ff in `find $SYN_DIR -iname "*.xml" | tail -n $NUM_TEST`;
   do
   fbase=`basename $ff .xml`
   name=$IM_DIR/$(echo $ff | cut -f 3,4 -d '/' | cut -f 1 -d '.').JPEG
   if [[ -a ${name} ]]; then
	   echo $fbase >> $SET_TEST
   fi
   done
done
cd -
