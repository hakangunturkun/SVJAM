from PIL import Image
from PIL import ImageFile
import numpy as np
import glob
import os
import sys
ImageFile.LOAD_TRUNCATED_IMAGES=True

#path = '/home/hakan/Documents/survivor_chr19/images_all_acf_512/'


'''
def files(path):
    for file in os.listdir(path):
        if not os.path.isfile(os.path.join(path, file)):
            yield file
file_ls=[]
for file in files(path):
    file_ls.append(file)
'''

def convert_size(path):
    for file in os.listdir(path):
        print(file)
        name_of_folder = str(file)
        files_in_folder=path+name_of_folder+'/'
        for trainimage in os.listdir(files_in_folder):
            trainimage1 = files_in_folder+str(trainimage)
            img = Image.open(trainimage1)
            img.load()
            #data = np.asarray( img, dtype="int32" )   
            if img.size == (512,512):
                continue   
            new_img = img.resize((512, 512))
            save_loc = path+name_of_folder+'/'+str(trainimage)
            new_img.save(save_loc)

'''	    
# single folder	    
for trainimage in os.listdir(path):
    trainimage1 = path+str(trainimage)
    img = Image.open(trainimage1)
    img.load()
    data = np.asarray( img, dtype="int32" )      
    new_img = img.resize((512, 512))
    save_loc = path+str(trainimage)
    new_img.save(save_loc)
'''
  
