#!/bin/bash
var=$(pwd)
cd $var

echo loading input
date +"%T"
xrdcp -r -s /afs/cern.ch/work/a/agoetz/public/lenv $var
xrdcp -s /afs/cern.ch/work/a/agoetz/public/SRWscripts/1c.py $var
echo imput loaded
date +"%T"


echo starting py
. $var/lenv/bin/activate
python3 $var/1c.py $2
deactivate
echo py finished
date +"%T"

echo starting transfer
xrdcp -r -s $var/./*.p /eos/home-a/agoetz/tempresults2/
echo transfer finished
date +"%T"
