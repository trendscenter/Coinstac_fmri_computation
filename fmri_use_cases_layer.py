#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This layer runs the pre-processing fmri (Voxel Based Morphometry) pipeline based on the inputs from interface adapter layer
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


import sys, os, glob, shutil, math, base64, warnings, getopt, re,traceback
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
import ujson as json

# Load bids layout interface for parsing bids data to extract T1w scans,subject names etc.
from bids import BIDSLayout

import nibabel as nib
import nipype.pipeline.engine as pe
import numpy as np
from nilearn import plotting

import fmri_entities_layer

#Stop printing nipype.workflow info to stdout
from nipype import logging
logging.getLogger('nipype.workflow').setLevel('CRITICAL')


def setup_pipeline(data='', write_dir='', data_type=None, **template_dict):
    """setup the pre-processing pipeline on T1W scans
        Args:
            data (array) : Input data
            write_dir (string): Directory to write outputs
            data_type (string): BIDS, niftis, dicoms
            template_dict ( dictionary) : Dictionary that stores all the paths, file names, software locations
        Returns:
            computation_output (json): {"output": {
                                                  "success": {
                                                    "type": "boolean"
                                                  },
                                                   "message": {
                                                    "type": "string",
                                                  },
                                                   "download_outputs": {
                                                    "type": "string",
                                                  },
                                                   "display": {
                                                    "type": "string",
                                                  }
                                                  }
                                        }
        Comments:
            After setting up the pipeline here , the pipeline is run with run_pipeline function
        """
    try:
        # Create pipeline nodes from fmri_entities_layer.py and pass them run_pipeline function
        [realign, slicetiming, datasink, fmri_preprocess] = create_pipeline_nodes(
            **template_dict)
        if data_type == 'bids':
            # Runs the pipeline on each subject serially
            layout = BIDSLayout(data)
            smri_data = layout.get(
                datatype='func', extensions='.nii.gz')
            return run_pipeline(
                write_dir,
                smri_data,
                realign,
                slicetiming,
                datasink,
                fmri_preprocess,
                data_type='bids',
                **template_dict)
        elif data_type == 'nifti':
            # Runs the pipeline on each nifti file serially
            smri_data = data
            return run_pipeline(
                write_dir,
                smri_data,
                realign,
                slicetiming,
                datasink,
                fmri_preprocess,
                data_type='nifti',
                **template_dict)
        elif data_type == 'dicoms':
            # Runs the pipeline on each nifti file serially
            smri_data = data
            return run_pipeline(
                write_dir,
                smri_data,
                realign,
                slicetiming,
                datasink,
                fmri_preprocess,
                data_type='dicoms',
                **template_dict)
    except Exception as e:
        sys.stdout.write(
            json.dumps({
                "output": {
                    "message": str(e)
                },
                "cache": {},
                "success": True
            }))


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


def write_readme_files(write_dir='', data_type=None, log=None,**template_dict):
    """This function writes readme files"""

    # Write a text file with info. on each of the output nifti files
    if data_type == 'bids':
        with open(
                os.path.join(write_dir, template_dict['outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['bids_outputs_manual_content'])
            fp.close()
    elif data_type == 'nifti':
        with open(
                os.path.join(write_dir, template_dict['outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['nifti_outputs_manual_content'])
            fp.close()
    elif data_type == 'dicoms':
        with open(
                os.path.join(write_dir, template_dict['outputs_manual_name']),
                'w') as fp:
            fp.write(template_dict['dicoms_outputs_manual_content'])
            fp.close()

    # Write a text file with info. on quality control correlation coefficent
    with open(os.path.join(write_dir, template_dict['qc_readme_name']),
              'w') as fp:
        fp.write(template_dict['qc_readme_content'])
        fp.close()

    # Write log of fmri output
    with open(os.path.join(write_dir, template_dict['log_filename']),
              'w') as fp:
        fp.write(log)
        fp.close()


def calculate_FD(rp_text_file,write_dir, sub_id,**template_dict):
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
    realignment_parameters[:, rot_indices] = rdist
    diff = np.diff(realignment_parameters, axis=0)
    FD_rms = np.sqrt(np.sum(diff**2, axis=1))
    FD_rms_mean = np.mean(FD_rms)
    write_path = os.path.dirname(rp_text_file)

    with open(
            os.path.join(write_path, template_dict['fmri_qc_filename']),
            'w') as fp:
        fp.write("%3.2f\n" % (FD_rms_mean))
        fp.close()

    #Flag subjects with >0.2 FD value
    if round(FD_rms_mean,2) > template_dict['FD_rms_mean_threshold']:
        with open(
                os.path.join(write_dir, template_dict['qa_flagged_filename']),
                'w') as fp:
            fp.write("%s\n" % (sub_id))
            fp.close()
    return FD_rms_mean

def nii_to_image_converter(write_dir, label, **template_dict):
    """This function converts nifti to base64 string"""
    import nibabel as nib
    from nilearn import plotting, image
    import os, base64

    file = glob.glob(os.path.join(write_dir, template_dict['display_nifti']))
    # mask = nib.load(file[0])
    mask = image.index_img(file[0], int(
        (image.load_img(file[0]).shape[3]) / 2))
    new_data = mask.get_data()

    clipped_img = nib.Nifti1Image(new_data, mask.affine, mask.header)

    plotting.plot_anat(
        clipped_img,
        cut_coords=(0, 0, 0),
        annotate=False,
        draw_cross=False,
        output_file=os.path.join(write_dir,
                                 template_dict['display_image_name']),
        display_mode='ortho',
        title=label + ' ' + template_dict['display_pngimage_name'],
        colorbar=False)

def resample_nifti_images(image_file, voxel_dimensions, resample_method):
    """Resample the NIfTI images in a folder and put them in a new folder
    Args:
        images_location: Path where the images are stored
        voxel_dimension: tuple (dx, dy, dz)
        resample_method: NN - Nearest neighbor
                         Li - Linear interpolation
    Returns:
        None:
    """
    try:
        voxel_size_str = '_{:.0f}mm'.format(float(voxel_dimensions[0]))
        (file_name, file_ext) = os.path.splitext(image_file)
        new_file_name = ''.join([file_name, voxel_size_str, file_ext])

        resample = afni.Resample()
        resample.inputs.environ = {'AFNI_NIFTI_TYPE_WARN': 'NO'}
        resample.inputs.in_file = image_file
        resample.inputs.out_file = os.path.join(os.path.dirname(image_file), new_file_name)
        resample.inputs.voxel_size = voxel_dimensions
        resample.inputs.outputtype = 'NIFTI'
        resample.inputs.resample_mode = resample_method
        resample.run()

        #Delete the image_file as we only use the resampled image
        if os.path.exists(image_file):os.remove(image_file)

    except Exception as e:
        sys.stderr.write('Unable to resample regression input file Error_log:' + str(e)+str(traceback.format_exc()))

    return os.path.join(os.path.dirname(image_file), new_file_name)


def create_pipeline_nodes(**template_dict):
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

    ## 6 Create the pipeline/workflow and connect the nodes created above ##
    fmri_preprocess = pe.Workflow(name="fmri_preprocess")

    fmri_preprocess.connect([
        create_workflow_input(
            source=realign.node,
            target=normalize.node,
            source_output='mean_image',
            target_input='image_to_align'),
        create_workflow_input(
            source=slicetiming.node,
            target=normalize.node,
            source_output='timecorrected_files',
            target_input='apply_to_files'),
        create_workflow_input(
            source=normalize.node,
            target=smooth.node,
            source_output='normalized_files',
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
            source_output='normalized_files',
            target_input=template_dict['fmri_output_dirname'] + '.@4'),
        create_workflow_input(
            source=smooth.node,
            target=datasink.node,
            source_output='smoothed_files',
            target_input=template_dict['fmri_output_dirname'] + '.@5')
    ])
    return [realign, slicetiming, datasink, fmri_preprocess]

def create_workflow_input(source, target, source_output, target_input):
    """This function collects pipeline nodes and their connections
    and returns them in appropriate format for nipype pipeline workflow
    """
    return (source, target, [(source_output, target_input)])

def convert_and_run_reorient_script(input_file):
    from pathlib2 import Path
    import shutil
    from nipype.interfaces import spm
    shutil.copy('/computation/reorient_template.m', '/computation/reorient.m')
    path = Path('/computation/reorient.m')
    text = path.read_text()
    text = text.replace('input_file', input_file)
    path.write_text(text)
    # Run convert_to_mat_file.m script using spm12 standalone and Matlab MCR
    with stdchannel_redirected(sys.stderr, os.devnull):
        spm.SPMCommand.set_mlab_paths(matlab_cmd='/opt/spm12/run_spm12.sh /opt/mcr/v95 script /computation/reorient_job.m',
                                  use_mcr=True)

def smooth_images(write_dir,**template_dict):
    """This function runs smoothing on input images. Ex: modulated images"""
    from nipype.interfaces import spm
    from nipype.interfaces.io import DataSink
    smooth = pe.Node(interface=spm.Smooth(), name='smooth')
    smooth.inputs.in_files = glob.glob(os.path.join(write_dir, 'wa*.nii'))
    smooth.inputs.fwhm = template_dict['FWHM_SMOOTH']
    fmri_smooth_modulated_images = pe.Workflow(
        name="fmri_smooth_modulated_images")
    datasink = pe.Node(interface=DataSink(), name='datasink')
    datasink.inputs.base_directory = write_dir
    fmri_smooth_modulated_images.connect([(smooth, datasink, [('smoothed_files',
                                                              write_dir)])])
    with stdchannel_redirected(sys.stderr, os.devnull):
        fmri_smooth_modulated_images.run()


def run_pipeline(write_dir,
                 smri_data,
                 realign,
                 slicetiming,
                 datasink,
                 fmri_preprocess,
                 data_type=None,
                 **template_dict):
    """This function runs pipeline"""

    unwanted_indexes = list()  # list to store indices of subjects which do not pass QA
    outputDirectory = write_dir
    id = 0  # id for assigning sub-id incase of nifti files in txt format
    loop_counter = 0  # loop counter
    count_success = 0  # variable for counting how many subjects were successfully run

    # Create regression_input_files to store input files for performing regression
    regression_input_dir=write_dir + '/' + template_dict[
        'regression_dir_name']
    os.makedirs(regression_input_dir,exist_ok=True)

    write_dir = write_dir + '/' + template_dict[
        'output_zip_dir']  # Store outputs in this directory for zipping the directory
    error_log = dict()  # dict for storing error log

    for each_sub in smri_data:
        loop_counter += 1

        try:

            # Assign subject,session id and input nifiti file for reorienation node
            if data_type == 'bids':
                sub_id = 'sub-' + each_sub.entities['subject']
                if 'session' in each_sub.entities:
                    session = each_sub.entities['session']
                else:
                    session = ''
                nii_output = ((
                    each_sub.filename).split('/')[-1]).split('.gz')[0]
                with stdchannel_redirected(sys.stderr, os.devnull):
                    n1_img = nib.load(each_sub.filename)

            if data_type == 'nifti':
                id = id + 1
                sub_id = 'subID-' + str(id)
                session = ''
                nii_output = ((each_sub).split('/')[-1]).split('.gz')[0]
                with stdchannel_redirected(sys.stderr, os.devnull):
                    n1_img = nib.load(each_sub)

            if data_type == 'dicoms':
                id = id + 1
                sub_id = 'subID-' + str(id)
                session = ''
                fmri_out = os.path.join(write_dir, sub_id, session, 'func')
                os.makedirs(fmri_out, exist_ok=True)

                ## This code runs the dicom to nifti conversion here
                from nipype.interfaces.dcm2nii import Dcm2niix
                dcm_nii_convert =  Dcm2niix()
                dcm_nii_convert.inputs.source_dir = each_sub
                dcm_nii_convert.inputs.output_dir = fmri_out
                with stdchannel_redirected(sys.stderr, os.devnull):
                    dcm_nii_convert.run()
                with stdchannel_redirected(sys.stderr, os.devnull):
                    n1_img = nib.load(glob.glob(os.path.join(fmri_out, '*.nii*'))[0])
                    nii_output=((glob.glob(os.path.join(fmri_out, '*.nii*'))[0]).split('/')[-1]).split('.gz')[0]

            # Directory in which fmri outputs will be written
            fmri_out = os.path.join(write_dir, sub_id, session, 'func')

            # Create output dir for sub_id
            os.makedirs(fmri_out, exist_ok=True)

            if n1_img:
                """
                Save nifti file from input data into output directory only if data_type !=dicoms because the dcm_nii_convert in the previous
                step saves the nifti file to output directory
                 """
                nib.save(n1_img, os.path.join(fmri_out, nii_output))

                # Create fmri_spm12 dir under the specific sub-id/func
                os.makedirs(
                    os.path.join(fmri_out, template_dict['fmri_output_dirname']),
                    exist_ok=True)

                nifti_file = glob.glob(os.path.join(fmri_out, '*.nii'))[0]

                # run reorientation node and pass to realign
                try:
                    with stdchannel_redirected(sys.stderr, os.devnull):
                        convert_and_run_reorient_script(nifti_file)
                except:
                    pass

                # Edit realign node inputs
                realign.node.inputs.in_files = nifti_file
                #realign.node.inputs.out_file = fmri_out + "/" + template_dict['fmri_output_dirname'] + "/Re.nii"
                #realign.node.run()

                if template_dict['options_slicetime_ref_slice'] is not None: slicetiming.node.inputs.ref_slice = template_dict[
                    'options_slicetime_ref_slice']


                # Edit Slicetiming node inputs
                TR = n1_img.header.get_zooms()[-1]
                if template_dict['options_repetition_time'] is not None: TR = template_dict['options_repetition_time']
                num_slices = n1_img.shape[2]
                slicetiming.node.inputs.in_files = nifti_file

                slicetiming.node.inputs.num_slices = num_slices
                if template_dict['options_num_slices'] is not None: slicetiming.node.inputs.num_slices = template_dict[
                    'options_num_slices']
                slicetiming.node.inputs.time_repetition = TR
                time_for_one_slice = TR / num_slices
                slicetiming.node.inputs.time_acquisition = TR - time_for_one_slice
                odd = range(1, num_slices + 1, 2)
                even = range(2, num_slices + 1, 2)
                acq_order = list(odd) + list(even)

                if template_dict['options_acquisition_order'] is not None: acq_order = template_dict[
                    'options_acquisition_order']
                slicetiming.node.inputs.slice_order = acq_order


                # Edit datasink node inputs
                datasink.node.inputs.base_directory = fmri_out

                # Run the nipype pipeline
                with stdchannel_redirected(sys.stderr, os.devnull):
                    fmri_preprocess.run()

                # Motion quality control: Calculate Framewise Displacement
                FD_rms_mean=calculate_FD(glob.glob(os.path.join(fmri_out,
                             template_dict['fmri_output_dirname'],'rp*.txt'))[0],write_dir,sub_id,**template_dict)


                # # Rename wmean*nii and swmean*nii to wa*nii and swa*nii files. This is done due to align the naming convention to spm12 normalizing naming convention
                # wmean_filename = ((glob.glob(os.path.join(fmri_out, template_dict['fmri_output_dirname'], 'wmean*.nii'))[0]).split('/'))[-1]
                # swmean_filename = ((glob.glob(os.path.join(fmri_out, template_dict['fmri_output_dirname'], 'swmean*.nii'))[0]).split('/'))[-1]
                # new_wmean_filename = (wmean_filename.split('mean'))[0] + 'a' + (wmean_filename.split('mean'))[1]
                # new_swmean_filename = (swmean_filename.split('mean'))[0] + 'a' + (swmean_filename.split('mean'))[1]
                # shutil.move(os.path.join(fmri_out, template_dict['fmri_output_dirname'], wmean_filename),os.path.join(fmri_out, template_dict['fmri_output_dirname'], new_wmean_filename))
                # shutil.move(os.path.join(fmri_out, template_dict['fmri_output_dirname'], swmean_filename),os.path.join(fmri_out, template_dict['fmri_output_dirname'], new_swmean_filename))


                label = sub_id + session
                with stdchannel_redirected(sys.stderr, os.devnull):
                    nii_to_image_converter(
                        os.path.join(fmri_out,
                                     template_dict['fmri_output_dirname']), label,
                        **template_dict)


        except Exception as e:
            # If the above code fails for any reason update the error log for the subject id
            # ex: the nifti file is not a nifti file
            # the input file is not a brian scan
            error_log.update({sub_id: str(e)+str(traceback.format_exc())})
            unwanted_indexes.append(loop_counter)
            continue

        else:

            # If the try block succeeds, increase the  success count and save the wc1*nii as wc1.png
            count_success = count_success + 1

            if count_success == 1:
                shutil.copy(
                    os.path.join(fmri_out, template_dict['fmri_output_dirname'],
                                 template_dict['display_image_name']),
                    os.path.dirname(write_dir))

            # Copy regression input files to regression_input_dir
            shutil.copy(os.path.join(glob.glob(
                os.path.join(fmri_out, template_dict['fmri_output_dirname'],
                             template_dict['regression_file_input_type'] + '*.nii'))[0]),
                        os.path.join(regression_input_dir,
                                     sub_id + session + '_' + template_dict['regression_file_input_type'] + '.nii'))

            if template_dict['regression_resample_voxel_size'] is not None:
                # Resample regression file input images for performing regression (for demo purposes)
                regression_resampled_file = resample_nifti_images(os.path.join(regression_input_dir,
                                                                               sub_id + session + '_' + template_dict[
                                                                                   'regression_file_input_type'] + '.nii'),
                                                                  template_dict['regression_resample_voxel_size'],
                                                                  template_dict['regression_resample_method'])

            if round(FD_rms_mean,2) > template_dict['FD_rms_mean_threshold']: unwanted_indexes.append(loop_counter)

            regression_resampled_file=glob.glob(os.path.join(regression_input_dir,sub_id + session + '_' + template_dict['regression_file_input_type'] + '.nii'))[0]

            template_dict['covariates'][0][0][loop_counter][0] = (regression_resampled_file).replace(outputDirectory+'/','')
            template_dict['regression_data'][0][loop_counter-1] = (regression_resampled_file).replace(outputDirectory + '/','')

        finally:
            remove_tmp_files()

    template_dict['covariates'][0][0]=[v for i, v in enumerate(template_dict['covariates'][0][0]) if i not in unwanted_indexes]
    template_dict['regression_data'][0] = [v for i, v in enumerate(template_dict['regression_data'][0]) if
                                         i not in [b-1 for b in unwanted_indexes] ]


    if os.path.isfile(
            os.path.join(
                os.path.dirname(write_dir),
                template_dict['display_image_name'])):
        #Zip output files
        shutil.make_archive(
            os.path.join(
                os.path.dirname(write_dir), template_dict['output_zip_dir']),
            'zip', write_dir)

        #Remove fmri_outputs directory if needed
        #shutil.rmtree(write_dir, ignore_errors=True)

        download_outputs_path = write_dir + '.zip'

        output_message = "fMRI preprocessing completed. Download zipped output file here:" +download_outputs_path+" " +str(
            count_success) + "/" + str(
                len(smri_data)
            ) + " subjects completed successfully." + template_dict[
                'coinstac_display_info']

        preprocessed_percentage = (count_success / len(smri_data)) * 100

        # If preprocessed_percentage<=template_dict['qc_threshold'] output qa warning
        if os.path.isfile(
                os.path.join(write_dir, template_dict['qa_flagged_filename'])):
            qa_percentage = (len(
                open(
                    os.path.join(write_dir,
                                 template_dict['qa_flagged_filename'])).
                readlines()) / len(smri_data)) * 100
            if (qa_percentage <= template_dict['qc_threshold']) or (preprocessed_percentage <= template_dict['qc_threshold']):
                output_message = output_message + template_dict['flag_warning']
        else:
            if (preprocessed_percentage <= template_dict['qc_threshold']):
                output_message = output_message + template_dict['flag_warning']

        if bool(error_log):
            output_message = output_message + " Error log:" + str(error_log)

        # Write readme files
        write_readme_files(write_dir, data_type, output_message, **template_dict)

        if preprocessed_percentage>template_dict['qc_threshold']:
            return json.dumps({
                "output": {
                    "covariates":template_dict['covariates'],
                    "data":template_dict['regression_data']
                },
                "cache": {},
                "success": True
            })
        else:
            return json.dumps({
                "output": {
                    "message": output_message
                },
                "cache": {},
                "success": True
            })
    else:
        return json.dumps({
            "output": {
                "message": "None of the input data could be pre-processed. Please check the data!"
            },
            "cache": {},
            "success": True
        })

