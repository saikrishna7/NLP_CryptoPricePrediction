
import os, zipfile

# Create a Lambda function
file_name = 'skaf48_s3-twitter-to-es-python.zip'
dir_name = 'skaf48_s3-twitter-to-es-python'


# os.chdir(dir_name) # change directory from working dir to dir with files

# folder_path = os.path.abspath(dir_name) # get full path of files
# folder_path
zip_ref = zipfile.ZipFile(file_name) # create zipfile object
zip_ref.extractall(dir_name) # extract file to dir
zip_ref.close() # close file