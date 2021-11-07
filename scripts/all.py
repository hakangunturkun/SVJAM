import sys
import os
import shutil
import glob
import scripts.gvcf_maker
import scripts.size_converter_in_folders
import scripts.crop_pca_cluster_detection
import scripts.cnn_predict

# create images_512 folder with 512*512 images for CNN 
def create_512(chrname):
    try:
        shutil.copytree('./Example/'+chrname+'/images', './Example/'+chrname+'/images_512/') 
    except FileExistsError:
        print('Exists: ./Example/'+chrname+'/images_512/')
    scripts.size_converter_in_folders.convert_size('./Example/'+chrname+'/images_512/')

# run clustering algorithm, check for already done    
def joint_anlys(chrname):
    for region in os.listdir('./Example/'+chrname+'/images'):
        if 'chr' in region and os.path.isdir('./Example/'+chrname+'/images/'+region):
            loc = region.split('_'+chrname)[0]
            if not (os.path.exists('./Example/'+chrname+'/images/'+loc+'_'+loc+'.csv') or os.path.exists('./Example/'+chrname+'/analysis/deletion/'+loc+'_'+loc+'.csv')):
                print('Running joint_calling for:', region)
                scripts.crop_pca_cluster_detection.joint_analysis('./Example/'+chrname+'/images/'+region,loc, './Example/'+chrname+'/images/')
            elif (os.path.exists('./Example/'+chrname+'/images/'+loc+'_'+loc+'.csv')):
                print('Exists: ./Example/'+chrname+'/images/'+loc+'_'+loc+'.csv' )
            else: 
                print('Exists: ./Example/'+chrname+'/analysis/deletion/'+loc+'_'+loc+'.csv')

# make necessary folders
def mkdirs(chrname):
    os.makedirs(os.path.dirname('./Example/'+chrname+'/analysis/'), exist_ok=True)
    os.makedirs(os.path.dirname('./Example/'+chrname+'/analysis/deletion/'), exist_ok=True)
    os.makedirs(os.path.dirname('./Example/'+chrname+'/analysis/del/'), exist_ok=True)
    os.makedirs(os.path.dirname('./Example/'+chrname+'/analysis/del/csv_log_files/'), exist_ok=True)
    os.makedirs(os.path.dirname('./Example/'+chrname+'/analysis/dup_inv/'), exist_ok=True)
    os.makedirs(os.path.dirname('./Example/'+chrname+'/analysis/dup_inv/csv_log_files/'), exist_ok=True)
    os.makedirs(os.path.dirname('./Example/'+chrname+'/analysis/dup_inv/pred_csv/'), exist_ok=True)

# copy results of clustering to related folders
def cp_mv(chrname):
    csvfiles = glob.iglob(os.path.join('./Example/'+chrname+'/images/', "*.csv"))
    for f1 in csvfiles:
        if os.path.isfile(f1):
            shutil.copy2(f1, './Example/'+chrname+'/analysis/del/csv_log_files/')
            shutil.copy2(f1, './Example/'+chrname+'/analysis/dup_inv/csv_log_files/')
            shutil.move(f1, './Example/'+chrname+'/analysis/deletion/')
            
    logfiles = glob.iglob(os.path.join('./Example/'+chrname+'/images/', "*.log"))
    for f2 in logfiles:
        if os.path.isfile(f2):
            shutil.copy2(f2, './Example/'+chrname+'/analysis/del/csv_log_files/')
            shutil.copy2(f2, './Example/'+chrname+'/analysis/dup_inv/csv_log_files/')
            shutil.move(f2, './Example/'+chrname+'/analysis/deletion/')
            
    pngfiles = glob.iglob(os.path.join('./Example/'+chrname+'/images/', "*.png"))
    for f3 in pngfiles:
        shutil.move(f3, './Example/'+chrname+'/analysis/deletion/')
# run CNN
def cnn_pred(chrname):
    scripts.cnn_predict.prediction('./Example/'+chrname+'/images_512/', './Example/'+chrname+'/analysis/dup_inv/csv_log_files/', './Example/'+chrname+'/analysis/dup_inv/pred_csv/')

# run gvcf_maker
def make_gvcf(chrname):
    scripts.gvcf_maker.implement(chrname, './Example/'+chrname+'/', './tools/' )

def attach_header(chrname):
    filenames = ['./scripts/header.txt', './Example/'+chrname+'/'+chrname+'_without_header.gvcf']
    with open('./Example/'+chrname+'/'+chrname+'.gvcf', 'w') as outputfile:
        for f in filenames:
            with open(f) as inputfile:
                for line in inputfile:
                    outputfile.write(line)

def combine_all(chrname):
    create_512(chrname)
    joint_anlys(chrname)
    mkdirs(chrname)
    cp_mv(chrname)
    cnn_pred(chrname)
    make_gvcf(chrname)
    attach_header(chrname)
