#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This layer defines the nodes of fmri pre-processing pipeline
"""

import nipype.pipeline.engine as pe
import nipype.interfaces.spm as spm
from nipype.interfaces.io import DataSink

#Stop printing nipype.workflow info to stdout
from nipype import logging
logging.getLogger('nipype.workflow').setLevel('CRITICAL')


## 1 Reorientation node & settings ##
class Realign:
    def __init__(self, **template_dict):
        self.node = pe.Node(interface=spm.Realign(), name='realign')
        self.node.inputs.paths = template_dict['spm_path']
        self.node.inputs.register_to_mean = False


## 2 Slicetiming Node and settings ##
class Slicetiming:
    def __init__(self, **template_dict):
        self.node = pe.Node(interface=spm.SliceTiming(), name='slicetiming')
        self.node.inputs.paths = template_dict['spm_path']


## 3 Normalize Node and settings ##
class Normalize:
    def __init__(self, **template_dict):
        self.node = pe.Node(interface=spm.Normalize12(), name='normalize')
        self.node.inputs.tpm = template_dict['tpm_path']


## 4 Smoothing Node & Settings ##
class Smooth:
    def __init__(self, **template_dict):
        self.node = pe.Node(interface=spm.Smooth(), name='smoothing')
        self.node.inputs.paths = template_dict['spm_path']
        self.node.inputs.fwhm = template_dict['FWHM_SMOOTH']


## 5 Datsink Node that collects segmented, smoothed files and writes to temp_write_dir ##
class Datasink:
    def __init__(self):
        self.node = pe.Node(interface=DataSink(), name='sinker')
