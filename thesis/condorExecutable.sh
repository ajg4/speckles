#!/bin/bash
var=$(pwd)
cd $var

echo loading input
date +"%T"
xrdcp -r -s /afs/cern.ch/work/a/agoetz/public/env.tar.gz $var
xrdcp -s /afs/cern.ch/work/a/agoetz/public/speckles/thesis/8_5.py $var
xrdcp -s /afs/cern.ch/work/a/agoetz/public/speckles/thesis/8_6.py $var
echo imput loaded
date +"%T"


echo starting py
tar -xzvf env.tar.gz
. $var/env/bin/activate
python3 $var/8_5.py
python3 $var/8_6.py $2
deactivate
echo py finished
date +"%T"

echo starting transfer
xrdcp -r -s $var/./*.p /eos/home-a/agoetz/tempresults2/
echo transfer finished
date +"%T"
