""" This script runs fmri pipeline on BIDS anatomical data using spm12 standalone and Matlab common runtime"""
# """Example run of the code- python3 /computation/run_fmri_bids.py '{"inputBidsDir":"/computation/test_bids_input_data","tempWriteDir":"/computation","SmoothingValue":[6, 6, 6]}'"""
""" Input args: --run json (this json structure may involve different field for different run) """
""" output: json """

## script name: run_fmri_bids.py ##
## import dependent libraries ##
import glob, os, sys, json, argparse, shutil, ast
import nibabel as nib

## Load Nipype interfaces ##
from nipype.interfaces import spm
from nipype.interfaces.spm import Realign,SliceTiming,Normalize12,Smooth
from nipype.interfaces.io import DataSink
from nipype.interfaces.utility import Function
import nipype.pipeline.engine as pe

# Load pybids packages to easily query the fmri tasks in bids data
from bids.grabbids import BIDSLayout


smooth_mm_value = [6, 6, 6]
# Set the paths to the SPM12 Template and transform.mat
transf_mat_path = '/computation/transform.mat'
spm_path=["/opt/spm12/fsroot"]
tpm_path = '/opt/spm12/fsroot/spm/spm12/tpm/TPM.nii'

if __name__=='__main__':

    doc = json.loads(sys.argv[1])
    input_bids_dir=doc['state']['baseDirectory']+'/test_bids_input_data'
    temp_write_dir='/output'
    
    layout=BIDSLayout(input_bids_dir)

#Check if number of fmri tasks in bids data is atleast 1
    if (len(layout.get_tasks()) > 0) and os.access(temp_write_dir, os.W_OK):

        # Get the paths to the fmri files to run the algorithm
        glob_str = os.path.join(input_bids_dir, 'sub*', 'func', '*.nii.gz')
        fmri_data = glob.glob(glob_str)

        ## Loop through each of the fmri files to run the algorithm, this algorithm runs serially ##
        i = 0  # variable for looping
        count_success = 0  # variable for counting how many subjects were successfully run
        
        #create dirs array to store output directories where fmri spm12 data is written to
        dirs=[]
        
        #create swafiles array to store paths to swa files for each subject
        swafiles=[]
        
        while i < len(fmri_data):
            gzip_file_path = fmri_data[i]
            i = i + 1

            # Extract subject directory name from the T1w*.nii.gz files
            sub_path = (os.path.dirname(os.path.dirname(gzip_file_path))).split('/')
            sub_id='/'.join(sub_path[2:len(sub_path)])

            fmri_out = temp_write_dir + '/' + sub_id + '/func'
            nii_output = (gzip_file_path.split('/')[-1]).split('.gz')[0]

            # Create output dir for sub_id
            os.makedirs(fmri_out, exist_ok=True)

            nifti_file = os.path.join(fmri_out, nii_output)
            img = nib.load(gzip_file_path)
            nib.save(img, os.path.join(fmri_out, nii_output))
            
            

            # Connect spm12 standalone to nipype
            matlab_cmd = '/opt/spm12/run_spm12.sh /opt/mcr/v92 script'
            spm.SPMCommand.set_mlab_paths(matlab_cmd=matlab_cmd, use_mcr=True)

            # Create fmri_spm12 dir under the specific sub-id/fun
            os.makedirs(os.path.join(fmri_out + "/fmri_spm12"), exist_ok=True)

            # Create the fmri pipeline using Nipype
            ## 1 Realign node and settings ##
            realign = Realign()
            if os.path.exists(nifti_file): realign.inputs.in_files = nifti_file
            realign.inputs.register_to_mean =False
            realign.inputs.paths=spm_path
            realign.run()
                	

            ## 2 Slice time correction Node and settings ##
            slicetiming = SliceTiming()
            TR=img.header.get_zooms()[-1]
            num_slices=img.shape[2]
            slicetiming.inputs.in_files =nifti_file
            slicetiming.inputs.num_slices = num_slices
            slicetiming.inputs.time_repetition = TR
            time_for_one_slice = TR / num_slices
            slicetiming.inputs.time_acquisition = TR - time_for_one_slice
            odd = range(1, num_slices+1, 2)
            even = range(2, num_slices+1, 2)
            acq_order = list(odd) + list(even)
            slicetiming.inputs.slice_order = acq_order
            slicetiming.inputs.ref_slice = int(num_slices/2)
            slicetiming.inputs.paths=spm_path
            slicetiming.run()
            

            ## 3 Normalizing Node & Settings ##
            normalize = Normalize12()
            normalize.inputs.tpm='/opt/spm12/fsroot/spm/spm12/tpm/TPM.nii'
            

            ## 4 Smoothing Node & Settings ##
            smoothing = Smooth()
            smoothing.inputs.fwhm = smooth_mm_value
            smoothing.inputs.paths=spm_path
     
            try:
                ## Remove any tmp files in the docker ##
                if (os.path.exists('/var/tmp')):
                    shutil.rmtree('/var/tmp', ignore_errors=True)

                for c in glob.glob(os.getcwd() + '/crash*'):
                    os.remove(c)

                for f in glob.glob(os.getcwd() + '/tmp*'):
                    shutil.rmtree(f, ignore_errors=True)

                for f in glob.glob(os.getcwd() + '/__pycache__'):
                    shutil.rmtree(f, ignore_errors=True)

                if os.path.exists(os.getcwd() + '/fmri_preprocess'):
                    shutil.rmtree(os.getcwd() + '/fmri_preprocess', ignore_errors=True)

                if os.path.exists(os.getcwd() + '/pyscript.m'):
                    os.remove(os.getcwd() + '/pyscript.m')

                ## Pipeline execution starts here.. ##

                # Run the fmri pipeline
                if os.path.exists(nifti_file):
                	normalize.inputs.image_to_align =glob.glob(fmri_out+'/mean*'+nii_output)
                	normalize.inputs.apply_to_files =glob.glob(fmri_out+'/a*'+nii_output)
                	normalize.run()
                	smoothing.inputs.in_files =glob.glob(fmri_out+'/wa*'+nii_output)
                	smoothing.run()
                    dirs.append(os.path.join(fmri_out + "/fmri_spm12"))
                    swafiles.append(glob.glob(fmri_out + "/fmri_spm12/swa*nii")[0])




                status = True
                #sys.stdout.write(json.dumps({"output": {"success": status,"fmridirs":dirs,"swafiles":swafiles},"success":status}))
            except Exception as e:
                # If fails raise the exception,set status False,write json output and print exception err string to std.err
                status = True
                sys.stderr.write(str(e))
                continue

            else:
                # If the try block succeeds, increase the count
                count_success = count_success + 1

            finally:
                ## Remove any tmp files in the docker ##
                if (os.path.exists('/var/tmp')): shutil.rmtree('/var/tmp', ignore_errors=True)
                for c in glob.glob(os.getcwd() + '/crash*'): os.remove(c)
                for f in glob.glob(os.getcwd() + '/tmp*'): shutil.rmtree(f, ignore_errors=True)
                for f in glob.glob(os.getcwd() + '/__pycache__'): shutil.rmtree(f, ignore_errors=True)
                if os.path.exists(os.getcwd() + '/fmri_preprocess'): shutil.rmtree(os.getcwd() + '/fmri_preprocess',
                                                                                  ignore_errors=True)
                if os.path.exists(os.getcwd() + '/pyscript.m'): os.remove(os.getcwd() + '/pyscript.m')

                # On the last subject in the BIDS directory , write the success status output to json object
                if gzip_file_path == fmri_data[-1]:
                    #if count_success > 0: status = True  # If atleast 1 scan in the BIDS directory finishes successfully
                    status = True
                    sys.stdout.write(json.dumps({"output": {"success": status,"vbmdirs": dirs,"swafiles":swafiles},"success":status}))


    # If input_bids_dir is not in BIDS format and does not have fmri data and no write permissions to tmp write dir then
    # Set the Status to False, write the error message to stderr and output the json object on stdout
    else:
        status = False
        sys.stderr.write(
            "Make sure data is in BIDS format,fmri data exists and space is available on the system to write outputs")
        sys.stdout.write(json.dumps({"output": {"success": status,"vbmdirs": dirs,"swafiles":swafiles},"success":status}))
