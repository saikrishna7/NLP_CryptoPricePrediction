import zipfile
import sys
import os

file_name = 'skaf48_s3-rssfeed-to-es-python.zip'
dir_name = 'skaf48_s3-rssfeed-to-es-python'

# Follow the link instead to zip the folder contents - https://unix.stackexchange.com/questions/182032/zip-the-contents-of-a-folder-without-including-the-folder-itself


def zip_folder(zf, folder):
    """Zip the contents of an entire folder (with that folder included
    in the archive). Empty subfolders will be included in the archive
    as well.
    """
    parent_folder = os.path.dirname(folder)
    # Retrieve the paths of the folder contents.
    contents = os.walk(folder)
    try:
        for root, folders, files in contents:
            # Include all subfolders, including empty ones.
            for folder_name in folders:
                absolute_path = os.path.join(root, folder_name)
                relative_path = absolute_path.replace(parent_folder + '\\',
                                                      '')
                print("Adding '%s' to archive." % absolute_path)
                zf.write(absolute_path, relative_path)
            for file_name in files:
                absolute_path = os.path.join(root, file_name)
                relative_path = absolute_path.replace(parent_folder + '\\',
                                                      '')
                print("Adding '%s' to archive." % absolute_path)
                zf.write(absolute_path, relative_path)
        print("'%s' created successfully." % zf)
    except(IOError, message):
        print(message)
        sys.exit(1)
    except(OSError, message):
        print(message)
        sys.exit(1)
    except(zf.BadZipfile, message):
        print(message)
        sys.exit(1)
    finally:
        zf.close()

        
# Functions defined,  get the party started:
zf = zipfile.ZipFile(dir_name+".zip", "w")
zip_folder(zf, dir_name)