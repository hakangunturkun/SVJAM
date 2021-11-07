import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as image
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import cv2
import sys
import re
import time

#import multiprocessing as mp
#from multiprocessing import Pool

from sklearn.decomposition import PCA

def joint_analysis(args1,args2, args3):
    ########################## Need to change here ################
    
    #loc = 'chr4_112361825-112865466'
    #path = ./images/'
    #
    # fill the first column of np.array to concatenate it with the other columns
    # hence specify one sample (fill the first column with this)
    sample_image="4512-JFI-0341_BXD149_loupe"
    
    #three types of images
    #type1= images here
    ###type2 = different chrs
    #type3= 1d images
    
    #choose the type please:
    
    #image_type = 1
    
    alt = "4512-JFI-0334_DBA_2J_loupe"
    ref = "4512-JFI-0333_C57BL_6J_loupe"
    
    ####################################################
    
    deletion_files=[]
    ls_sum_len =[]
    
    path=args1 + "/"
    loc=args2
    folder_path=args3
    
    old_stdout = sys.stdout
    log_file = open(folder_path+loc + '_' + loc + '.log',"w")
    sys.stdout = log_file
    
    #print ("reading "+path+ loc + '_' + loc + '_' + sample_image + '.png')
    print(path+ loc + '_' + loc + '_' + sample_image + '.png')
    
    '''
    img = cv2.imread(path+ loc + '_' + loc + '_' + sample_image + '.png',0)
    crop_img = img[0:1024,144:1168]
    for i in range(2,1021):
        crop_img[i][1025-i]=255
        crop_img[i][1024-i]=255
        crop_img[i][1023-i]=255
        crop_img[i][1022-i]=255
        crop_img[i][1021-i]=255
    img_all_sample = (crop_img / 255).reshape(-1,1)
    img_all = np.array(img_all_sample)
    '''
    
    #filename_ls = [sample_image]
    
    #filename_long = [loc + '_' + loc + '_' + sample_image + '.png']
    
    filename_ls=[]
    filename_long=[]
    t=0
    
    for filename in os.listdir(path):
        if not filename.endswith(".png"):
            continue
        t+=1
        #if filename != path+loc + '_' + loc + '_' + sample_image + '.png':
        #grayscale = Image.open(path+filename).convert('L')
        #print(filename)
        filename_long.append(filename)
        img = cv2.imread(path+filename,0)
        crop_img = img[0:1024,144:1168]    
        for i in range(2,1021):
            crop_img[i][1025-i]=255
            crop_img[i][1024-i]=255
            crop_img[i][1023-i]=255
            crop_img[i][1022-i]=255
            crop_img[i][1021-i]=255
        img_r = (crop_img / 255.0).reshape(-1,1)
        if (t==1):
            img_all=np.array(img_r)
        else:
            img_all = np.concatenate((img_all,img_r),axis=1)
        filename1 = filename.split(loc + '_' + loc + '_')[1]
        filename2 = filename1.split(".png")[0]
        filename_ls.append(filename2)
        #filename3 = re.search('_(.+?)_loupe',filename2)
        #filename_gvcf.append(filename3.group(1))
        ls_sum=[]
        
        
        '''
        # find the sum of all elements in rows, make them a list ls_sum
        def multi_pro(i):
            sumr=0
            for j in range(1024):
                sumr += crop_img[j,i]
            return(sumr)
            
        #for k in range(1025-dup_size):
        if __name__ == '__main__':
            with Pool(8) as p:
                #print(p.map(deneme(k)))
                ls_sum.append(p.map(multi_pro,range(1024)))
                
        '''
        # find the sum of all elements in rows, make them a list ls_sum
        sumr=0
        for i in range(1024):
            sumr=0
            for j in range(1024):
                sumr += crop_img[j,i]    
            ls_sum.append(sumr)
        
            
        # find the deletions, append the number of them in a list ls_sum_len
        # append the filenames with deletions to deletion_files
        if(261120 in ls_sum):
            ls_sum_len.append(ls_sum.count(261120))
            deletion_files.append(filename2)
                
    #filename_ls.remove(sample_image)
    #filename_long.remove(loc + '_' + loc + '_' + sample_image + '.png')
    #print(filename_ls)
    
    '''
    img = cv2.imread(path+ loc + '_' + loc + '_' + sample_image + '.png',0)
    crop_img = img[0:1024,144:1168]
    for i in range(2,1021):
        crop_img[i][1025-i]=255
        crop_img[i][1024-i]=255
        crop_img[i][1023-i]=255
        crop_img[i][1022-i]=255
        crop_img[i][1021-i]=255
    img_all_sample = (crop_img / 255).reshape(-1,1)
    img_all=np.delete((img_all_sample,img_r),axis=1)
    
    '''
    if (deletion_files==[]):
        print('There are no deletions in this region')
    elif(len(deletion_files)==1):
        print('There is only one file with deletion in this region:', deletion_files)
    else:
        if(len(deletion_files)>=150):
            print('ALL the files have deletion in this region.')
        # find the outlier boundary
        out_val = int((sorted(ls_sum_len)[int(len(ls_sum_len)*(3/4))]-sorted(ls_sum_len)[int(len(ls_sum_len)*(1/4))])*(3/2)+sorted(ls_sum_len)[int(len(ls_sum_len)*(3/4))])
        
        num_len_sv=0
        for len_sv in ls_sum_len:
            if len_sv == out_val:
                num_len_sv += 1
        if(num_len_sv >= int(len(ls_sum_len)*(1/2))):
            sample_elt=out_val
        else:
            # take the closest element to the boundary, hence remove the mistakes or de novo deletions
            sample_elt = max([i for i in ls_sum_len if out_val>i])
        sample_index = ls_sum_len.index(sample_elt)
    
        # take the filename of this element, make it the sample for deletion detection
        del_detect_file = deletion_files[sample_index]
       
        sample_image1 = del_detect_file
        img = cv2.imread(path+ loc + '_' + loc + '_' + sample_image1 + '.png',0)
        crop_img = img[0:1024,144:1168]
        for i in range(2,1021):
                crop_img[i][1025-i]=255
                crop_img[i][1024-i]=255
                crop_img[i][1023-i]=255
                crop_img[i][1022-i]=255
                crop_img[i][1021-i]=255
        ls_sum=[]
        sumr=0
        for i in range(1024):
            sumr=0
            for j in range(1024):
                sumr += crop_img[j,i]    
            ls_sum.append(sumr)
        
        li=[]
        for i,j in enumerate(ls_sum):
            if(j==261120):
                li.append(i)
        
        # extract the begin and end of the location from the loc
        loc_begin = str.split(str.split(loc,'_')[1],'-')[0]
        loc_end = str.split(str.split(loc,'_')[1],'-')[1]
    
        # find the interval size 
        loc_length = int(loc_end)-int(loc_begin)
    
        # find the number of bases for each pixel
        magnify = loc_length/1024
        
        # set sv_size
        sv_size = 1000
        
        # set the number of pixels that may be noisy in between two white rows
        skip_size = 5000/magnify
    
        # set minimum sv_size to detect in terms of pixels, here it 5000 bp
        min_sv_size = sv_size/magnify
    
        # find the pixels having deletion and find the intervals
        def find_del_interval(k):
            global li_sep 
            li_sep = []
            li_sep.append(li[k-1])
            if (len(li)==2 and li[1]-li[0]<skip_size):
                li_sep = li
                return(2)
            else:
                for i in range(k,len(li)):
                    if i==0 or i==1:
                        continue
    	        # if there are pixels with a few nonwhite colors, 
    	        # ignore them if the number of them are less than min_sv_size
                    elif li[i]-li[i-1]<skip_size:
                        li_sep.append(li[i])
                        if(i == len(li)-1):
                            #print(li_sep)
                            return(i+1)
                    else:
                        #print(k)
                        #li_sep.append(li[i-1])
                        k = i+1
                        #print(li_sep)
                        return(k)
                        break
                
        t=0
        i=1
        if (len(li)==2 and li[i]-li[i-1]>=skip_size):
            print('We cannot detect a deletion with less than ', sv_size, ':', deletion_files)
        else:
            while i<len(li):
        	    t = find_del_interval(i)
        	    i=t
        	    #print(li_sep[-1]-li_sep[0])
        	    #print(min_sv_size)
        	    if(len(li_sep)>1 and (li_sep[-1]-li_sep[0]>min_sv_size)):
        	        print('Deletion interval in pixels:[',li_sep[0],',',li_sep[-1],']')
        	        print('Deletion interval in bp:[',int(int(loc_begin)+magnify*int(li_sep[0])),',',int(int(loc_begin)+magnify*int(li_sep[-1])),']')
        	    else:
        	    	print('We cannot detect a deletion with less than ', sv_size)
    
    
    def rotation(angle, matrix):
        global df_np
        rotate = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]]) 
        matrix_tr = np.transpose(matrix)
        matrix_tr=np.matmul(rotate,matrix_tr)
        matrix = np.transpose(matrix_tr)
        df_k = pd.DataFrame(matrix,columns=['x','y'])
        df_np = np.array(df_k)
        return df_np
    
    df = pd.DataFrame(img_all, columns=filename_ls)
    X_std = StandardScaler().fit_transform(df)
    X_tr = np.transpose(X_std)
    pca = PCA(n_components=2)
    prin2 = pca.fit_transform(X_tr)
    prin2_tr = np.transpose(prin2)
    
    
    #angle = (100*np.pi)/(180)
    #rotate = np.array([[np.cos(angle),-np.sin(angle)], [np.sin(angle),np.cos(angle)]])
    #prin2_tr = np.matmul(rotate,prin2_tr)
    
    #fig=plt.figure()
    #ax=fig.add_axes([0,0,1,1])
    #ax.scatter(prin2_tr[0], prin2_tr[1], color='r')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_title('scatter plot')
    #plt.show()
    
    df_k = pd.DataFrame(prin2,columns=['x','y'])
    
    df_graph = pd.DataFrame({
    'x': df_k['x'],
    'y': df_k['y'],
    'group': filename_ls
    })
    
    p1=sns.regplot(data=df_graph, x="x", y="y", fit_reg=False, marker="o", color="skyblue", scatter_kws={'s':10})
    for line in range(0,df_graph.shape[0]):
         p1.text(df_graph.x[line]+0.2, df_graph.y[line], df_graph.group[line], horizontalalignment='left', size='3', color='black', weight='semibold')
    figure = p1.get_figure()    
    figure.savefig(folder_path+loc + '_' + loc + '_' + 'pca.png', dpi=400)
    
    def within_distance(name, matrix, labels, cluster):
        within_total=0
        w = df_temp.iloc[df_temp.index[df_temp['name'] == name].tolist()[0],1]
        name_index = df_temp.index[df_temp['name'] == name][0]    
        label_w = np.where(labels==w)[0]
        for i in range(len(label_w)):
            within_total += matrix[name_index,label_w[i]]
        within_mean = within_total/(len(label_w))
        return within_mean
    
    def between_distance(name, matrix, labels, cluster):
        between_total=0
        label_0 = np.where(labels==0)[0]
        label_1 = np.where(labels==1)[0]
        label_2 = np.where(labels==2)[0]
        
        w = df_temp.iloc[df_temp.index[df_temp['name'] == name].tolist()[0],1]
        name_index = df_temp.index[df_temp['name'] == name][0]
        label_w = np.where(labels==w)[0]
        
        if(w==0):    
            for i in range(len(label_1)):
                between_total += matrix[name_index,label_1[i]]
            for i in range(len(label_2)):
                between_total += matrix[name_index,label_2[i]]
            between_mean = between_total/(len(label_1)+len(label_2))
        elif(w==1):    
            for i in range(len(label_0)):
                between_total += matrix[name_index,label_0[i]]
            for i in range(len(label_2)):
                between_total += matrix[name_index,label_2[i]]
            between_mean = between_total/(len(label_0)+len(label_2))
        else:    
            for i in range(len(label_0)):
                between_total += matrix[name_index,label_0[i]]
            for i in range(len(label_1)):
                between_total += matrix[name_index,label_1[i]]
            between_mean = between_total/(len(label_0)+len(label_1))
        return between_mean
    
    def prob(name, matrix, labels):
        between_total=0
        label_0 = np.where(labels==0)[0]
        label_1 = np.where(labels==1)[0]
        label_2 = np.where(labels==2)[0]
        
        label_C57BL = df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1]
        label_DBA_2J = df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1]
    
        dist_to_DBA_2J = matrix[df_temp.index[df_temp['name'] == name][0],df_temp.index[df_temp['name'] == alt][0]]
        dist_to_C57BL = matrix[df_temp.index[df_temp['name'] == name][0],df_temp.index[df_temp['name'] == ref][0]]    
        
        w = df_temp.iloc[df_temp.index[df_temp['name'] == name].tolist()[0],1]
        name_index = df_temp.index[df_temp['name'] == name][0]
        label_w = np.where(labels==w)[0]
        #np.where(labels== (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1])[0])):
        dist_dba2j = dist_to_DBA_2J/mean_dist_ref_DBA_2J
        dist_c57bl = dist_to_C57BL/mean_dist_ref_C57BL
        dist_het = abs((dist_c57bl-dist_dba2j)/np.sqrt(2))
        if (dist_c57bl+dist_dba2j+dist_het) == 0:
            dist_c57bl =0.01
        if(w==label_DBA_2J):
            proba_to_DBA_2J = 1-(dist_dba2j)/(dist_c57bl+dist_dba2j+dist_het)
            proba_to_C57BL = (1-proba_to_DBA_2J)*dist_het/(dist_c57bl+dist_het)
            proba_to_HET = (1-proba_to_DBA_2J)*(dist_c57bl)/(dist_c57bl+dist_het)   
        elif(w==label_C57BL):    
            proba_to_C57BL = 1-(dist_c57bl)/(dist_c57bl+dist_dba2j+dist_het)
            proba_to_DBA_2J = (1-proba_to_C57BL)*dist_het/(dist_dba2j+dist_het)
            proba_to_HET = (1-proba_to_C57BL)*dist_dba2j/(dist_dba2j+dist_het)
        else:
            proba_to_HET  = 1-(dist_het)/(dist_c57bl+dist_dba2j+dist_het)
            proba_to_DBA_2J = (1-proba_to_HET)*(dist_c57bl)/(dist_dba2j+dist_c57bl)
            proba_to_C57BL = (1-proba_to_HET)*(dist_dba2j)/(dist_dba2j+dist_c57bl)
             
        return (float("{:.3f}".format(proba_to_DBA_2J)),float("{:.3f}".format(proba_to_HET)), float("{:.3f}".format(proba_to_C57BL)))
    
    def distance_to_centers(name, labels,puns):
        label_C57BL = df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1]
        label_DBA_2J = df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1]
        
        label_ref = np.where(labels==label_C57BL)
        label_alt = np.where(labels==label_DBA_2J)
           
        centroid_ref=(0.0)
        sum1=(0.0,0.0)
        for i in label_ref[0]:
            sum1 = np.array(sum1)+np.array(df_np[i])
        centroid_ref = sum1/len(label_ref[0])
        
        centroid_alt=(0.0)
        sum2=(0.0,0.0)
        for i in label_alt[0]:
            sum2 = np.array(sum2)+np.array(df_np[i])
        centroid_alt = sum2/len(label_alt[0])
    
        global dist_of_alt_ref_centers
        dist_of_alt_ref_centers = round(puns*abs(centroid_ref[0]-centroid_alt[0])+abs(centroid_ref[1]-centroid_alt[1]),2)
        
        dist_to_centroid_ref_x = puns*abs(df_np[df_temp.index[df_temp['name'] == name]][0][0] - centroid_ref[0])
        dist_to_centroid_ref_y = abs(df_np[df_temp.index[df_temp['name'] == name]][0][1] - centroid_ref[1])
        
        dist_to_centroid_alt_x = puns*abs(df_np[df_temp.index[df_temp['name'] == name]][0][0] - centroid_alt[0])
        dist_to_centroid_alt_y = abs(df_np[df_temp.index[df_temp['name'] == name]][0][1] - centroid_alt[1])
      
        dist_to_centroid_ref = round((dist_to_centroid_ref_x + dist_to_centroid_ref_y),2)
        dist_to_centroid_alt = round((dist_to_centroid_alt_x + dist_to_centroid_alt_y),2)
            
        return(dist_to_centroid_ref, dist_to_centroid_alt)
    
    def mean_dist_of_hets_to_centers(labels,puns):
        label_C57BL = df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1]
        label_DBA_2J = df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1]
        
        label_ref = np.where(labels==label_C57BL)
        label_alt = np.where(labels==label_DBA_2J)
             
        label_hets = set([0, 1, 2]) - set([label_C57BL, label_DBA_2J])  
        temp_list = list(label_hets)
        het_label = temp_list[0]
        
        label_het = np.where(labels==het_label)
        
        sum1=(0.0,0.0)
        for i in label_het[0]:
            #print(i)
            #print(df_temp.iloc[i]['name'])
            #print(distance_to_centers(df_temp.iloc[i]['name'], labels,puns))
            sum1 = np.array(sum1)+np.array(distance_to_centers(df_temp.iloc[i]['name'], labels,puns))
        aver = sum1/len(label_het[0])        
        return(aver)
    
    def total_distance_to_ref(name, ref, matrix):
        total_dist_ref=0
        name_index = df_temp.index[df_temp['name'] == name][0]  
        ref_index = df_temp.index[df_temp['name'] == ref][0]
        total_dist_ref = matrix[name_index,ref_index]
        return total_dist_ref
    
    df_np = np.array(df_k)
    matr = np.zeros(shape=(df_k.shape[0],df_k.shape[0]))
    
    quality = [0]
    angles = [0]
    
    for angle_degree in range(0,200,5):
        angle = (angle_degree*np.pi)/(180)
        df_k = pd.DataFrame(prin2,columns=['x','y'])
        df_np = np.array(df_k)
        df_np=rotation(angle, df_np)
        #cluster_metrics =[]
        for i in range(df_k.shape[0]):
            for j in range(df_k.shape[0]):
                matr[i,j] = (5*abs(df_np[i,0]-df_np[j,0])+abs(df_np[i,1]-df_np[j,1]))
        clt = AgglomerativeClustering(linkage='single', affinity='precomputed', n_clusters=3)
        model = clt.fit(matr)
        labels = clt.labels_
        #cluster_metrics.append(metrics.calinski_harabasz_score(matr, labels))  
        df_temp = pd.DataFrame(data = {'name' : filename_ls,  'label': clt.labels_})
        total_within = 0 
        for name in df_temp['name']:
            total_within += within_distance(name, matr,clt.labels_,3)
        mean_within = (total_within) / df_temp.shape[0]
    
        total_between = 0 
        for name in df_temp['name']:
            total_between += between_distance(name, matr,clt.labels_,3)
        mean_between = (total_between) / df_temp.shape[0]
    
        #if ((df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] != df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1]) and (1-mean_within/(mean_between-mean_within)<1)):
        if (1-mean_within/(mean_between-mean_within)<1):
            quality.append(1-mean_within/(mean_between+mean_within))
            angles.append(angle_degree)
    
    try:
        best_angle = angles[np.argmax(quality)]
        print("The best angle of rotation is: ", best_angle)
    except:
        best_angle = 0
        print("There is a problem in clustering")
    
    angle = (best_angle*np.pi)/(180)
    df_k = pd.DataFrame(prin2,columns=['x','y'])
    df_np = np.array(df_k)
            
    df_np=rotation(angle, df_np)
    
    # find best pnuishment weight if n_clusters=3
    mw=[1000]
    punish_mw =[0]
    temp_mean_within = 1
    
    for pun in range(3,15):
        for i in range(df_k.shape[0]):
            for j in range(df_k.shape[0]):
                matr[i,j] = (pun*abs(df_np[i,0]-df_np[j,0])+abs(df_np[i,1]-df_np[j,1]))
    
        clt = AgglomerativeClustering(linkage='single', affinity='precomputed', n_clusters=3)
        model = clt.fit(matr)  
        labels = clt.labels_ 
    
        df_temp = pd.DataFrame(data = {'name' : filename_ls,  'label': clt.labels_})
        total_within = 0 
        for name in df_temp['name']:
            total_within += within_distance(name, matr,clt.labels_,3)
        mean_within = (total_within) / df_temp.shape[0]
        mw.append(round((mean_within-temp_mean_within),2))
        punish_mw.append(pun)
        temp_mean_within = mean_within
    
    # if there is is a bad cluster and changes to a good one, remove that 
    if(mw[np.argmin(mw)]<0):
        for u in range(1,10):
            if mw[np.argmin(mw)-(u+1)] !=0:
                if(mw[np.argmin(mw)-u]/mw[np.argmin(mw)-(u+1)]>2):
                    punish_mw.remove(punish_mw[np.argmin(mw)])
                    mw.remove(mw[np.argmin(mw)])
                    break
                else:
                    if(mw[np.argmin(mw)-u]/mw[np.argmin(mw)-(u+1)]==1):
                        continue 
                    else:
                        if(mw[np.argmin(mw)-(u+1)]/mw[np.argmin(mw)-(u+2)]>2):
                            punish_mw.remove(punish_mw[np.argmin(mw)])
                            mw.remove(mw[np.argmin(mw)])
                        break
        
    best_pun = punish_mw[np.argmin(mw)]
    
    
    #fill the matrix by using best punishment weight
    for i in range(df_k.shape[0]):
        for j in range(df_k.shape[0]):
            matr[i,j] = (best_pun*abs(df_np[i,0]-df_np[j,0])+abs(df_np[i,1]-df_np[j,1]))
    
    #find calinski_harabasz_score when n_clusters=2 and n_clusters=3
    cluster_metrics =[]
    for i in range(2,4):
        clt = AgglomerativeClustering(linkage='single', affinity='precomputed', n_clusters=i)
        model = clt.fit(matr)
        labels = clt.labels_
        cluster_metrics.append(metrics.calinski_harabasz_score(matr, labels))
    df_temp = pd.DataFrame(data = {'name' : filename_ls,  'label': clt.labels_})
    
    
    total_dist_centers = 0 
    for name in df_temp['name']:
        total_dist_centers += np.array(distance_to_centers(name, labels,best_pun))
    
    mean_dist_centers = (total_dist_centers) / df_temp.shape[0]
    if mean_dist_centers[0]== 0 and mean_dist_centers[1]==0:
        mean_dist_centers[0]= 0.01
        mean_dist_centers[1]= 0.01
    het_dist = mean_dist_of_hets_to_centers(labels,best_pun)/mean_dist_centers
    print("Mean distance of hets to the other clusters (ref, alt):", het_dist)
    if het_dist[0] ==0 and het_dist[1]==0:
        het_dist[0]=0.01
        het_dist[1]=0.01
    print("\n Distance ratios:", het_dist[0]/het_dist[1], het_dist[1]/het_dist[0])
    
    
    #if (max(cluster_metrics)<120 or (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1])):
    if (max(cluster_metrics)<120):
        print("FAILED: " + loc)
        cls = 1
    elif(het_dist[0]/het_dist[1] > 2.5 or het_dist[1]/het_dist[0]>2.5 or mean_dist_of_hets_to_centers(labels,best_pun)[0]>dist_of_alt_ref_centers or mean_dist_of_hets_to_centers(labels,best_pun)[1]>dist_of_alt_ref_centers):
        print("\n Choose 2 clusters")
        cls = 2
    else:
        print("\n Choose 3 clusters")
        cls = 3    
    
    #if n_clusters=2, find best punishment weight
    if(cls==2):
        mw=[1000]
        punish_mw =[0]
        temp_mean_within = 1
        for pun in range(3,15):
            for i in range(df_k.shape[0]):
                for j in range(df_k.shape[0]):
                    matr[i,j] = (pun*abs(df_np[i,0]-df_np[j,0])+abs(df_np[i,1]-df_np[j,1]))
    
            clt = AgglomerativeClustering(linkage='single', affinity='precomputed', n_clusters=2)
            model = clt.fit(matr)  
            labels = clt.labels_ 
    
            df_temp = pd.DataFrame(data = {'name' : filename_ls,  'label': clt.labels_})
            total_within = 0 
            for name in df_temp['name']:
                total_within += within_distance(name, matr,clt.labels_,cls)
            mean_within = (total_within) / df_temp.shape[0]
            mw.append(round((mean_within-temp_mean_within),2))
            punish_mw.append(pun)
            temp_mean_within = mean_within
    
        if(mw[np.argmin(mw)]<0 and mw[np.argmin(mw)-1]/mw[np.argmin(mw)-2]>2):
            punish_mw.remove(punish_mw[np.argmin(mw)])
            mw.remove(mw[np.argmin(mw)])
    
    best_pun = punish_mw[np.argmin(mw)]
    print("best punishment:", best_pun)
    
    for i in range(df_k.shape[0]):
        for j in range(df_k.shape[0]):
            matr[i,j] = (best_pun*abs(df_np[i,0]-df_np[j,0])+abs(df_np[i,1]-df_np[j,1]))
    
    
    label_list =[]
    clt = AgglomerativeClustering(linkage='single', affinity='precomputed', n_clusters=cls)
    model = clt.fit(matr)  
    labels = clt.labels_
    label_list.append(labels)
    
    df_temp = pd.DataFrame(data = {'name' : filename_ls,  'label': clt.labels_})
    total_within = 0 
    for name in df_temp['name']:
        total_within += within_distance(name, matr,clt.labels_,i)
    mean_within = (total_within) / df_temp.shape[0]
    
    #if (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1]):
    #    print("FAILED: " + loc)
    #    cls = 1
    
    if (cls!=1):
        total_between = 0 
        for name in df_temp['name']:
            total_between += between_distance(name, matr,clt.labels_,i)
        mean_between = (total_between) / df_temp.shape[0]
        quality_measure = float("{:.3f}".format(1-mean_within/(mean_between+mean_within)))
        print("Quality of ", str(cls), " clusters: ", quality_measure)
    else:
        quality_measure = 0
    
    
    plt.scatter(df_k['x'], df_k['y'], c= model.labels_.astype(float), s=20, alpha=0.5)
    fig = plt.gcf()
    fig.savefig(folder_path+loc + '_' + loc + '_' +'cluster_' + str(cls) + '.png', dpi=400)
    
    fig_new = plt.figure()
    ax = fig_new.add_subplot(1, 1, 1)
    x=df_k['x']
    y=df_k['y']
    gr = df_graph['group']
    ax.scatter(x, y, alpha=0.8, c=model.labels_.astype(float), edgecolors='none', s=20, label=df_graph['group'])
    for line in range(0,df_k.shape[0]):
         ax.text(df_k.x[line]+0.3, df_k.y[line], df_graph.group[line], horizontalalignment='left', size='3', color='black', weight='normal') 
    fig_new.savefig(folder_path+loc + '_' + loc + '_' +'cluster_' + str(cls) +'_labelled.png', dpi=400)
    
    
    dist_vec_to_C57BL = matr[df_temp.index[df_temp['name'] == ref][0],]
    dist_vec_to_DBA_2J = matr[df_temp.index[df_temp['name'] == alt][0],]
    dist_of_C57BL_to_DBA_2J = matr[df_temp.index[df_temp['name'] == ref][0],df_temp.index[df_temp['name'] == alt][0]]
    
    
    total_dist_ref_C57BL = 0 
    total_dist_ref_DBA_2J = 0 
    for name in df_temp['name']:
        total_dist_ref_C57BL += total_distance_to_ref(name, ref, matr)
        total_dist_ref_DBA_2J += total_distance_to_ref(name, alt, matr)
    mean_dist_ref_C57BL = (total_dist_ref_C57BL) / df_temp.shape[0]
    mean_dist_ref_DBA_2J = (total_dist_ref_DBA_2J) / df_temp.shape[0]
    
    probalities_ref =[]
    probalities_alt =[]
    probalities_het =[]
    probalities = []
    
    label_C57BL = df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1]
    label_DBA_2J = df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1]
    
    for name in df_temp['name']:
        probalities_ref.append(prob(name, matr, clt.labels_)[2])
        probalities_alt.append(prob(name, matr, clt.labels_)[0])
        probalities_het.append(prob(name, matr, clt.labels_)[1])
        w = df_temp.iloc[df_temp.index[df_temp['name'] == name].tolist()[0],1]
        if(w==label_DBA_2J):
            probalities.append(prob(name, matr, clt.labels_)[0])
        elif(w==label_C57BL):
            probalities.append(prob(name, matr, clt.labels_)[2])
        else:
            probalities.append(prob(name, matr, clt.labels_)[1])
    
    df_gr_pr = pd.DataFrame({'x': df_k['x'], 'y': df_k['y'], 'prob': probalities})
    
    fig_prob = plt.figure()
    ax = fig_prob.add_subplot(1, 1, 1)
    x=df_gr_pr['x']
    y=df_gr_pr['y']
    prob = df_gr_pr['prob']
    ax.scatter(x, y, alpha=0.8, c=model.labels_.astype(float), edgecolors='none', s=30, label=df_gr_pr['prob'])
    for line in range(0,df_gr_pr.shape[0]):
         ax.text(df_gr_pr.x[line]+0.3, df_gr_pr.y[line], df_gr_pr.prob[line], horizontalalignment='left', size='3', color='black', weight='semibold') 
    fig_prob.savefig(folder_path+loc + '_' + loc + '_' +'cluster' + str(cls) +'_prob.png', dpi=400)
    
    
    df_prob = pd.DataFrame({'Distance_to_ref' : dist_vec_to_C57BL/mean_dist_ref_C57BL, 'Distance_to_alt' : dist_vec_to_DBA_2J/mean_dist_ref_DBA_2J, 'group' : filename_ls})
    
    figure_dist = plt.clf() 
    p_ref=sns.regplot(data=df_prob, x="Distance_to_ref", y="Distance_to_alt", fit_reg=False, marker="o", color="red", scatter_kws={'s':10})
    for line in range(0,df_prob.shape[0]):
         p_ref.text(df_prob.Distance_to_ref[line]+0.04, df_prob.Distance_to_alt[line], df_prob.group[line], horizontalalignment='left', size='3', color='brown', weight='regular') 
    figure_dist = p_ref.get_figure()  
    figure_dist.savefig(folder_path+loc + '_' + loc + '_' + 'distances_to_ref_alt.png', dpi=400)
    
    sys.stdout = old_stdout
    log_file.close()
    #if(cls==1):
    #    sys.exit()
    
    df_temp = pd.DataFrame(data = {'name' : filename_ls,  'label': clt.labels_})
    
    genotype = []
    label_cnn = []
    
    for i in label_list[0]:
        if ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 1) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 0)):
            if i == 0:
                genotype.append('1/1')
                label_cnn.append('1')
            elif i == 1:
                genotype.append('0/0')
                label_cnn.append('0')
            else:
                genotype.append('0/1')
                label_cnn.append('2')
        elif ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 0) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 1)): 
            if i == 0:
                genotype.append('0/0')
                label_cnn.append('0')
            elif i == 1:
                genotype.append('1/1')
                label_cnn.append('1')
            else:
                genotype.append('0/1')
                label_cnn.append('2')
        elif ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 2) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 0)): 
            if i == 0:
                genotype.append('1/1')
                label_cnn.append('1')
            elif i == 1:
                genotype.append('0/1')
                label_cnn.append('2')
            else:
                genotype.append('0/0')
                label_cnn.append('0')
        elif ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 0) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 2)): 
            if i == 0:
                genotype.append('0/0')
                label_cnn.append('0')
            elif i == 1:
                genotype.append('0/1')
                label_cnn.append('2')
            else:
                genotype.append('1/1')
                label_cnn.append('1')
        elif ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 1) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 2)): 
            if i == 0:
                genotype.append('0/1')
                label_cnn.append('2')
            elif i == 1:
                genotype.append('0/0')
                label_cnn.append('0')
            else:
                genotype.append('1/1')
                label_cnn.append('1')
        elif ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 2) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 1)): 
            if i == 0:
                genotype.append('0/1')
                label_cnn.append('2')
            elif i == 1:
                genotype.append('1/1')
                label_cnn.append('1')
            else:
                genotype.append('0/0')
                label_cnn.append('0')
        elif ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 0) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 0)): 
            if i == 0:
                genotype.append('0/0')
                label_cnn.append('0')
            elif i == 1:
                genotype.append('1/1')
                label_cnn.append('1')
            else:
                genotype.append('0/1')
                label_cnn.append('2')
        elif ((df_temp.iloc[df_temp.index[df_temp['name'] == ref].tolist()[0],1] == 1) and (df_temp.iloc[df_temp.index[df_temp['name'] == alt].tolist()[0],1] == 1)): 
            if i == 0:
                genotype.append('1/1')
                label_cnn.append('1')
            elif i == 1:
                genotype.append('0/0')
                label_cnn.append('0')
            else:
                genotype.append('0/1')
                label_cnn.append('2')
        else: 
            if i == 0:
                genotype.append('1/1')
                label_cnn.append('1')
            elif i == 1:
                genotype.append('0/0')
                label_cnn.append('0')
            else:
                genotype.append('0/1')
                label_cnn.append('2')
    
    df0 = pd.DataFrame(data = { 'genotype' : genotype, 'fullname' : filename_long, 'name' : filename_ls,  'label': label_cnn, 'prob_ref' : probalities_ref,'prob_het' : probalities_het, 'prob_alt' : probalities_alt, 'cls_qual' : quality_measure })
    df0.sort_values(by=["genotype"], ascending=False,inplace=True)
    df0.to_csv(folder_path+loc + '_' + loc + '.csv', index = False, sep='\t')
