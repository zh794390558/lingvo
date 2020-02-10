#!/usr/bin/env python3

import numpy
import sys
from kaldiio import WriteHelper

basedir=sys.argv[1]

numspk=5
utt2spk = open(basedir +'/data/utt2spk', 'w')

ark=basedir+'/data/feats.ark'
scp=basedir+'/data/feats.scp'

with WriteHelper('ark,scp:%s,%s' % (ark, scp)) as writer:
    for i in range(10):
        spkid = numpy.random.choice(numspk)
        spk = 'spk' + str(spkid) + '_'  # spk1_
        utt = spk + str(i)  # spk1_utt1
        writer(utt, numpy.random.randn(10, 10))
        utt2spk.write("%s %s\n" % (utt, spk)) 

utt2spk.close()
