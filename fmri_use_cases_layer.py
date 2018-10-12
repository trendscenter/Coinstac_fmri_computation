#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This layer runs the fmri pre-processing pipeline based on the inputs from interface adapter layer
This layer uses entities layer to modify nodes of the pipeline as needed
"""

import contextlib


@contextlib.contextmanager
def stdchannel_redirected(stdchannel, dest_filename):
    """
    A context manager to temporarily redirect stdout or stderr

    e.g.:


    with stdchannel_redirected(sys.stderr, os.devnull):
        if compiler.has_function('clock_gettime', libraries=['rt']):
            libraries.append('rt')
    """

    try:
        oldstdchannel = os.dup(stdchannel.fileno())
        dest_file = open(dest_filename, 'w')
        os.dup2(dest_file.fileno(), stdchannel.fileno())

        yield
    finally:
        if oldstdchannel is not None:
            os.dup2(oldstdchannel, stdchannel.fileno())
        if dest_file is not None:
            dest_file.close()


import sys, os, glob, shutil, math, base64, warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
import ujson as json
from bids.grabbids import BIDSLayout
import nibabel as nib
import nipype.pipeline.engine as pe
import numpy as np
from nilearn import plotting

import fmri_entities_layer

#Stop printing nipype.workflow info to stdout
from nipype import logging
logging.getLogger('nipype.workflow').setLevel('CRITICAL')


def execute_pipeline(bids_dir='',
                     write_dir='',
                     nii_files='',
                     data_type=None,
                     pipeline_opts=None,
                     **template_dict):
    """Runs the pre-processing pipeline on fmri scans in BIDS data
        Args:
            bids_dir (string) : Input BIds directory
            write_dir (string): Directory to write outputs
            pipeline_opts ( integer) : Options to change pipeline
            template_dict ( dictionary) : Dictionary that stores all the paths, file names, software locations
        Returns:
            computation_output (json): {"output": {
                                                  "success": {
                                                    "type": "boolean"
                                                  },
                                                   "fmridirs": {
                                                    "type": "array",
                                                    "contains": ["string"]
                                                  },
                                                  "swafiles": {
                                                    "type": "array",
                                                    "contains": ["string"]
                                                  }
                                                  }
                                        }
        Comments:
            After setting up the pipeline here , the pipeline is run

        """

    [realign, slicetiming, datasink, fmri_preprocess] = create_pipeline_nodes(
        pipeline_opts, **template_dict)

    if data_type == 'bids':
        # Runs the pipeline on each subject, this algorithm runs serially
        layout = BIDSLayout(bids_dir)
        fmri_data = layout.get(modality='func', extensions='.nii.gz')
        return run_pipeline(
            write_dir,
            fmri_data,
            realign,
            slicetiming,
            datasink,
            fmri_preprocess,
            data_type='bids',
            **template_dict)
    elif data_type == 'nifti':
        # Runs the pipeline on each nifti file, this algorithm runs serially
        fmri_data = nii_files
        return run_pipeline(
            write_dir,
            fmri_data,
            realign,
            slicetiming,
            datasink,
            fmri_preprocess,
            data_type='nifti',
            **template_dict)


def remove_tmp_files():
    """this function removes any tmp files in the docker"""

    for a in glob.glob('/var/tmp/*'):
        os.remove(a)

    for b in glob.glob(os.getcwd() + '/crash*'):
        os.remove(b)

    for c in glob.glob(os.getcwd() + '/tmp*'):
        shutil.rmtree(c, ignore_errors=True)

    for d in glob.glob(os.getcwd() + '/__pycache__'):
        shutil.rmtree(d, ignore_errors=True)

    shutil.rmtree(os.getcwd() + '/fmri_preprocess', ignore_errors=True)

    if os.path.exists(os.getcwd() + '/pyscript.m'):
        os.remove(os.getcwd() + '/pyscript.m')


def write_readme_files(write_dir='', data_type=None, **template_dict):
    """This function writes readme files"""

    # Write a text file with info. on each of the output nifti files
    if data_type == 'bids':
        with open(
                os.path.join(write_dir,
                             template_dict['bids_outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['bids_outputs_manual_content'])
            fp.close()
    elif data_type == 'nifti':
        with open(
                os.path.join(write_dir,
                             template_dict['nifti_outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['nifti_outputs_manual_content'])
            fp.close()

    # Write a text file with info. on quality control correlation coefficent
    with open(os.path.join(write_dir, template_dict['qc_readme_name']),
              'w') as fp:
        fp.write(template_dict['qc_readme_content'])
        fp.close()


def calculate_FD(rp_text_file,**template_dict):
    """Calculates Framewise displacement from realignment parameters. realignment parameters is calculated from realignment of raw nifti
            Args:
                realignment parameters.txt file
            Returns:
                Mean of RMS of Framewise displacement
            Comments:
                Framewise Displacement of a time series is defined as the sum of the absolute values of the derivatives of the six realignment parameters.
                realignmental displacements are converted from degrees to millimeters by calculating displacement on the surface of a sphere of radius 50 mm.
            """
    realignment_parameters = np.loadtxt(rp_text_file)
    rot_indices = range(3, 6)
    rad = 50
    # assume head radius of 50mm
    rot = realignment_parameters[:, rot_indices]
    rdist = rad * np.tan(rot)
    realignment_parameters[:, rot_ind] = rdist
    diff = np.diff(realignment_parameters, axis=0)
    FD_rms = np.sqrt(np.sum(diff**2, axis=1))
    FD_rms_mean = np.mean(FD_rms)
    write_path = os.path.dirname(rp_text_file)

    with open(
            os.path.join(write_path, template_dict['fmri_qc_filename']),
            'w') as fp:
        fp.write("%3.2f\n" % (FD_rms_mean))
        fp.close()


def nii_to_string_converter(input_dir, label, **template_dict):
    """This function converts nifti to base64 string"""
    import nibabel as nib
    from nilearn import plotting, image
    import os, base64

    file = glob.glob(os.path.join(input_dir, template_dict['display_nifti']))
    mask = image.index_img(file[0], int(
        (image.load_img(file[0]).shape[3]) / 2))
    new_data = mask.get_data()

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)

    plotting.plot_anat(
        clipped_img,
        cut_coords=(0, 0, 0),
        annotate=False,
        draw_cross=False,
        output_file=os.path.join(input_dir,
                                 template_dict['display_image_name']),
        display_mode='ortho',
        title=label + ' ' + template_dict['display_pngimage_name'],
        colorbar=False)


def create_pipeline_nodes(pipeline_opts, **template_dict):
    """This function creates and modifies nodes of the pipeline from entities layer with nipype

        smooth.node.inputs.fwhm: (a list of from 3 to 3 items which are a float or a float)
        3-list of fwhm for each dimension
        This is the size of the Gaussian (in mm) for smoothing the preprocessed data by. This is typically between about 4mm and 12mm.

    """

    # 1 Realign node and settings #
    realign = fmri_entities_layer.Realign(**template_dict)

    # 2 Slicetiming Node and settings #
    slicetiming = fmri_entities_layer.Slicetiming(**template_dict)

    # 3 Normalize Node and settings #
    normalize = fmri_entities_layer.Normalize(**template_dict)

    # 4 Smoothing Node & Settings #
    smooth = fmri_entities_layer.Smooth(**template_dict)

    # 5 Datsink Node that collects swa files and writes to temp_write_dir #
    datasink = fmri_entities_layer.Datasink()

    # 6 Modify Pipeline based on opts to update smoothing fwhm ( full width half maximum ) in mm in x,y,z directions
    if pipeline_opts is not None:
        fwhm = float(pipeline_opts)
        smooth.node.inputs.fwhm = [fwhm] * 3

    ## 6 Create the pipeline/workflow and connect the nodes created above ##
    fmri_preprocess = pe.Workflow(name="fmri_preprocess")

    fmri_preprocess.connect([
        create_workflow_input(
            source=realign.node,
            target=normalize.node,
            source_output='modified_in_files',
            target_input='apply_to_files'),
        create_workflow_input(
            source=slicetiming.node,
            target=normalize.node,
            source_output='timecorrected_files',
            target_input='image_to_align'),
        create_workflow_input(
            source=normalize.node,
            target=smooth.node,
            source_output='normalized_image',
            target_input='in_files'),
        create_workflow_input(
            source=realign.node,
            target=datasink.node,
            source_output='mean_image',
            target_input=template_dict['fmri_output_dirname']),
        create_workflow_input(
            source=realign.node,
            target=datasink.node,
            source_output='realigned_files',
            target_input=template_dict['fmri_output_dirname'] + '.@1'),
        create_workflow_input(
            source=realign.node,
            target=datasink.node,
            source_output='realignment_parameters',
            target_input=template_dict['fmri_output_dirname'] + '.@2'),
        create_workflow_input(
            source=slicetiming.node,
            target=datasink.node,
            source_output='timecorrected_files',
            target_input=template_dict['fmri_output_dirname'] + '.@3'),
        create_workflow_input(
            source=normalize.node,
            target=datasink.node,
            source_output='normalized_image',
            target_input=template_dict['fmri_output_dirname'] + '.@4'),
        create_workflow_input(
            source=smooth.node,
            target=datasink.node,
            source_output='smoothed_files',
            target_input=template_dict['fmri_output_dirname'] + '.@5')
    ])
    return [realign, slicetiming, datasink, fmri_preprocess]


def create_workflow_input(source, target, source_output, target_input):
    return (source, target, [(source_output, target_input)])


def run_pipeline(write_dir,
                 fmri_data,
                 realign,
                 slicetiming,
                 datasink,
                 fmri_preprocess,
                 data_type=None,
                 **template_dict):
    """This function runs pipeline on the current case"""

    id = 0  # id for assigning sub-id incase of nifti files in txt format
    count_success = 0  # variable for counting how many subjects were successfully run
    write_dir = write_dir + '/' + template_dict['output_zip_dir']  # Store outputs in this directory for zipping the directory
    error_log = dict()  # Dictionary for storing error log for each subject

    for each_sub in fmri_data:

        try:

            # Extract subject id and name of nifti file
            if data_type == 'bids':
                sub_id = 'sub-' + each_sub.subject
                session_id = getattr(each_sub, 'session', None)

                if session_id is not None:
                    session = 'ses-' + getattr(each_sub, 'session', None)
                else:
                    session = ''

                nii_output = ((
                    each_sub.filename).split('/')[-1]).split('.gz')[0]
                n1_img = nib.load(each_sub.filename)

            if data_type == 'nifti':
                id = id + 1
                sub_id = 'sub-' + str(id)
                nii_output = ((each_sub).split('/')[-1]).split('.gz')[0]
                n1_img = nib.load(each_sub)

            # Directory in which fmri outputs will be written
            fmri_out = os.path.join(write_dir, sub_id, session, 'func')

            # Create output dir for sub_id
            os.makedirs(fmri_out, exist_ok=True)

            nifti_file = os.path.join(fmri_out, nii_output)

            if n1_img:
                nib.save(n1_img, os.path.join(fmri_out, nii_output))

                # Create fmri_spm12 dir under the specific sub-id/func
                os.makedirs(
                    os.path.join(fmri_out,
                                 template_dict['fmri_output_dirname']),
                    exist_ok=True)

                # Edit realign node inputs
                realign.node.inputs.in_files = nifti_file
                #realign.node.inputs.out_file = fmri_out + "/" + template_dict['fmri_output_dirname'] + "/Re.nii"
                #realign.node.run()

                # Edit Slicetiming node inputs
                TR = n1_img.header.get_zooms()[-1]
                num_slices = n1_img.shape[2]
                slicetiming.node.inputs.in_files = nifti_file
                slicetiming.node.inputs.num_slices = num_slices
                slicetiming.node.inputs.time_repetition = TR
                time_for_one_slice = TR / num_slices
                slicetiming.node.inputs.time_acquisition = TR - time_for_one_slice
                odd = range(1, num_slices + 1, 2)
                even = range(2, num_slices + 1, 2)
                acq_order = list(odd) + list(even)
                slicetiming.node.inputs.slice_order = acq_order
                slicetiming.node.inputs.ref_slice = int(num_slices / 2)
                #slicetiming.node.run()

                # Edit datasink node inputs
                datasink.node.inputs.base_directory = fmri_out

                # Run the nipype pipeline
                with stdchannel_redirected(sys.stderr, os.devnull):
                    fmri_preprocess.run()

                # Motion quality control: Calculate Framewise Displacement
                calculate_FD(glob.glob(os.path.join(fmri_out,
                             template_dict['fmri_output_dirname'],'rp*.txt'))[0],**template_dict)

                # Write readme files
                write_readme_files(write_dir, data_type, **template_dict)

                label = sub_id + session
                nii_to_string_converter(
                    os.path.join(fmri_out,
                                 template_dict['fmri_output_dirname']), label,
                    **template_dict)

        except Exception as e:
            # If fails raise the exception,print exception error
            #sys.stderr.write(str(e))
            error_log.update({sub_id: str(e)})
            continue

        else:
            # If the try block succeeds, increase the count
            count_success = count_success + 1

            if count_success == 1:
                shutil.copy(
                    os.path.join(fmri_out,
                                 template_dict['fmri_output_dirname'],
                                 template_dict['display_image_name']),
                    os.path.dirname(write_dir))

        finally:
            remove_tmp_files()

    #Zip output files
    shutil.make_archive(
        os.path.join(
            os.path.dirname(write_dir), template_dict['output_zip_dir']),
        'zip', write_dir)

    #shutil.rmtree(write_dir, ignore_errors=True)
    '''
    Calculate how many nifti's successfully got run through the pipeline, this may help in colloborative projects
    where some of the projects may have low quality data
    '''

    download_outputs_path = write_dir + '.zip'

    output_message = "fMRI preprocessing completed. " + str(
        count_success) + "/" + str(
            len(fmri_data)) + " subjects" + " completed successfully."

    if bool(error_log):
        output_message = output_message + " Error log:" + str(error_log)

    with open(
            os.path.join(
                os.path.dirname(write_dir),
                template_dict['display_image_name']), "rb") as imageFile:
        encoded_image_str = base64.b64encode(imageFile.read())

    return json.dumps({
        "output": {
            "message": output_message,
            "download_outputs": download_outputs_path,
            "display": encoded_image_str
        },
        "cache": {},
        "success": True,
    })
