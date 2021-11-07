import sys
import os
import pandas as pd
import numpy as np
import shutil
import math
import re
import subprocess
from scipy import stats

def implement(args1, args2, args3):

    for sample_file in os.listdir(args2+'/analysis/del/csv_log_files/'):
        if sample_file.endswith(".csv"):
            break

    df_sample=pd.read_csv(args2+'/analysis/del/csv_log_files/'+sample_file,dtype=str, sep='\t')
    df_sample = df_sample.sort_values(by=['name'])

    col_list=['#CHROM', 'POS', 'CLS_QUAL', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
    name_list = df_sample['fullname'].str.split('.',expand=True)[0].str.split('_',expand=True)[4]+'_'+df_sample['fullname'].str.split('.',expand=True)[0].str.split('_',expand=True)[5]
    name_list = name_list.tolist()
    col_list = col_list + name_list
    df_gvcf=pd.DataFrame(columns=col_list)

    csv_li = []
    for filename in os.listdir(args2+'/analysis/del/csv_log_files/'):
        if filename.endswith('.csv'):
            df_csv = pd.read_csv(args2+'/analysis/del/csv_log_files/'+filename, index_col=None, sep='\t')
            df_csv = df_csv.sort_values(by=['name'])
            csv_li.append(df_csv)
    df0 = pd.concat(csv_li, axis=0, ignore_index=True)

    #with open(args3+'scripts/chr4_survivor_with_svtype.list') as sv_ls:
    #    svtype_ls = sv_ls.readlines()

    for filename in os.listdir(args2+'/analysis/del/csv_log_files/'):
        if filename.endswith('.log'):
            #print(filename)
            window_bgn = filename.split('_')[1].split('-')[0]
            window_end = filename.split('-')[1].split('_')[0]
            #print('window_bgn:',window_bgn)
            #print('window_end',window_end)
            window_middle_1 = ((int(window_end)-int(window_bgn))/5)*2+int(window_bgn)
            window_middle_2 = int(window_end)-((int(window_end)-int(window_bgn))/5)*2
            #for line_sv in svtype_ls:
            #    if (window_bgn in line_sv):
            #       sv_type_lr=line_sv.split('\t')[2]
            #print(filename, window_bgn, sv_type_lr)
            ## only longranger annotation
            #if str(sv_type_lr) != 'DEL\n':
                #print(filename, window_bgn, sv_type_lr )
                #continue
       
            with open(args2+'/analysis/del/csv_log_files/'+filename, 'r') as info:
                row1=[]
                # beginning_loc = []
                # ending_loc = []
                beginning_loc_all = []
                ending_loc_all = []
                svlen_all = []
                svlen = []
                alt_list=[]
                cls_qual=''
                failed_flag=0
                deletion_exists=0
     
                for line in info:
                
                    if line.startswith('Quality of '):
                        cls_qual = line.split(':  ')[1].split('\n')[0]
                    if line.startswith('Deletion interval in bp'):
                        # check if the deletion region is in the middle
                        if not ((int(line.split(':[ ')[1].split(' ,')[0])>window_middle_2) or (int(line.split(', ')[1].split(' ]')[0])<window_middle_1)):
                            #print("INSIDE:", filename, int(line.split(':[ ')[1].split(' ,')[0]),(int(line.split(', ')[1].split(' ]')[0])),window_middle_1,window_middle_2)
                        #else:
                        #    print("OUTSIDE:", filename, int(line.split(':[ ')[1].split(' ,')[0]),(int(line.split(', ')[1].split(' ]')[0])),window_middle_1,window_middle_2)
                            beginning_loc_all.append(line.split(':[ ')[1].split(' ,')[0])
                            ending_loc_all.append(line.split(', ')[1].split(' ]')[0])
                            svlen_all.append(int(line.split(', ')[1].split(' ]')[0])-int(line.split(':[ ')[1].split(' ,')[0]))
                            #alt_list.append('['+chrom_list[0]+':'+line.split(', ')[1].split(' ]')[0]+'[')
                            deletion_exists=1
                            #print(filename)
                            #print(line)
                    if (('FAILED' in line) or ('ALL' in line)):
                        failed_flag=1
                        continue
                        '''
                        #print('FAILED:', filename)
                        #chr_name = line.split(': ')[1].split('_')[0]
                        chr_name = str(filename).split('_')[0]
                        probstr = '1/1:1,0,0' 
                        probs=[]
                        for i in range(152):
                            probs.append(probstr)
                        cls_qual = '1'
                        '''
                # choose DEL with max svlen
                if len(svlen_all) !=0:
                    #print(filename,beginning_loc_all,ending_loc_all, svlen_all)
                    beginning_loc=beginning_loc_all[svlen_all.index(max(svlen_all))]
                    ending_loc=ending_loc_all[svlen_all.index(max(svlen_all))]
                    svlen=svlen_all[svlen_all.index(max(svlen_all))]
                else:
                    beginning_loc=0
                    ending_loc=0
                    svlen=0
                #print(filename,beginning_loc,ending_loc, svlen)
                if failed_flag==1:
                    continue
                if failed_flag==0 and deletion_exists==1:    
                    csv_name=args2+'/analysis/del/csv_log_files/'+str(filename).split('.')[0]+'.csv'
                    df_csv=pd.read_csv(csv_name,dtype=str, sep='\t')
                    df_csv = df_csv.sort_values(by=['name'])
                    chr_name = df_csv['fullname'].str.split('_',expand=True)[0][0]
                    probs = df_csv['genotype']+':'+df_csv['prob_ref'] + ',' +df_csv['prob_het'] + ',' +df_csv['prob_alt']
                if failed_flag==0 and deletion_exists==0:
                    continue
                #print(beginning_loc)
                #print(ending_loc)
                #print(svlen)
                #print(alt_list)
                #print(cls_qual)
            
                #for i in range(len(beginning_loc)):
                if beginning_loc != 0:
                    with open(args3+'test.bed', 'w') as tb:
                        tb.write(chr_name+"\t"+str(beginning_loc)+"\t"+str(int(beginning_loc)+1)+"\ta")
                        #tb.write("chr2\t14423253\t14423254\tab")
                    tb.close()
                    
                
                    #out = subprocess.run(['bedtools getfasta -fi UCSC_mm10.fa -bed test.bed'], shell=True,  capture_output=True)
                    out = subprocess.run(['bedtools getfasta -fi '+args3+'UCSC_mm10.fa -bed '+args3+'test.bed'], shell=True,  capture_output=True)
                    base = out.stdout.decode().split('\n')[1]
                    #print(filename)
                    #print(base)
                    if (round(stats.hmean([float(cls_qual),1]),3) >=0.75):
                        filter_score = 'PASS'
                    else:
                        filter_score = 'LOWQ'
                    row1=[chr_name, beginning_loc,cls_qual,'.', base , '<DEL>',round(stats.hmean([float(cls_qual),1]),3),filter_score ,'IMPRECISE;SVTYPE=DEL;SVEND='+ending_loc +';SVLEN=-'+str(svlen)+';CLS_QUAL='+cls_qual+';QUAL=1', 'GT:MP']
                    if failed_flag==0:
                        row1 = row1+probs.tolist()
                    if failed_flag==1:
                        row1 = row1+probs
                    #print((row1))
                    #print((df_gvcf))
                    df_gvcf.loc[len(df_gvcf)]=row1             
    df_gvcf.POS = df_gvcf.POS.astype(float)
    df_gvcf = df_gvcf.sort_values(['#CHROM', 'POS'])
    df_gvcf.POS = df_gvcf.POS.astype(int)
    df_gvcf = df_gvcf.reset_index(drop=True)           
                
    i=1
    last_index=len(df_gvcf['POS'])
    while i<last_index:
        if ((df_gvcf['#CHROM'][i]==df_gvcf['#CHROM'][i-1]) and (int(df_gvcf['POS'][i])-int(df_gvcf['POS'][i-1]))<1000):
            if df_gvcf['CLS_QUAL'][i]>df_gvcf['CLS_QUAL'][i-1]:
                df_gvcf=df_gvcf.drop(df_gvcf.index[[i-1]])
                i=1
                last_index = len(df_gvcf['POS'])
                df_gvcf = df_gvcf.reset_index(drop=True)
                continue
            else:
                df_gvcf=df_gvcf.drop(df_gvcf.index[[i]])    
                df_gvcf = df_gvcf.reset_index(drop=True)
                i=1
                last_index = len(df_gvcf['POS'])
                continue
        i+=1
    
    del df_gvcf['CLS_QUAL']
    
    
    ############### DUP ###########
    
    df_gvcf_dup=pd.DataFrame(columns=col_list)
    
    csv_li_dup = []
    for filename in os.listdir(args2+'/analysis/dup_inv/pred_csv/'):
        if filename.endswith('.csv'):
            df_csv_dup = pd.read_csv(args2+'/analysis/dup_inv/pred_csv/'+filename, index_col=None, sep=',')
            df_csv_dup = df_csv_dup.sort_values(by=['name'])
            csv_li_dup.append(df_csv_dup)
    df0_dup = pd.concat(csv_li_dup, axis=0, ignore_index=True)
    
    with open(args3+'sv_list_with_CI.list') as f_ci:
        ci_list = f_ci.readlines()
    
    k=0
    for filename in os.listdir(args2+'/analysis/dup_inv/pred_csv/'):
        #print(filename)
        window_bgn = filename.split('_')[1].split('-')[0]
        #for line_sv in svtype_ls:
        #    if (window_bgn in line_sv):
        #        sv_type_lr=line_sv.split('\t')[2]
        if filename.endswith('.csv'):    
            df_csv_dup=pd.read_csv(args2+'/analysis/dup_inv/pred_csv/'+filename,dtype=str, sep=',')
            df_csv_dup = df_csv_dup.sort_values(by=['name'])
            chr_name_dup = filename.split('_')[0]
            loc1_dup = filename.split('_')[1].split('-')[0]
            loc2_dup = filename.split('_')[1].split('-')[1].split('_')[0]       
            svlen_dup=(int(loc2_dup)-int(loc1_dup))/5
            beginning_loc_dup=int(loc1_dup)+2*int(svlen_dup)    
            ending_loc_dup=int(loc2_dup)-2*int(svlen_dup)
            cipos1=0
            cipos2=0
            ciend1=0
            ciend2=0
            for line_ci in ci_list:
                if ((chr_name_dup in line_ci) and (str(loc1_dup) in line_ci)):
                   cipos1=int(line_ci.split(':')[3])
                   cipos2=int(line_ci.split(':')[4])
                   ciend1=int(line_ci.split(':')[5])
                   ciend2=int(line_ci.split(':')[6])
            #if cipos1==0 and cipos2==0 and ciend1==0 and ciend2==0:
            #    continue
            cls_qual_dup = df_csv_dup['cls_qual'].iloc[0]    
            if(cls_qual_dup == '0'):
                cls_qual_dup = '1'    

            df_csv_dup=pd.read_csv(args2+'/analysis/dup_inv/pred_csv/'+filename,dtype=str, sep=',')
            df_csv_dup = df_csv_dup.sort_values(by=['name'])
            # subset of all with label==0:
            df_temp_0_dup=df_csv_dup[df_csv_dup['label'] == '0']
            if df_temp_0_dup.shape[0] != 0:
                # partitions of label 0 among predictions
                df_temp_00_dup=df_temp_0_dup[df_temp_0_dup['predictions']=='0.0']
                df_temp_01_dup=df_temp_0_dup[df_temp_0_dup['predictions']=='1.0']
                df_temp_02_dup=df_temp_0_dup[df_temp_0_dup['predictions']=='2.0']

                if ((df_temp_00_dup.shape[0]>df_temp_01_dup.shape[0]) and (df_temp_00_dup.shape[0]>df_temp_02_dup.shape[0])):
                    df_temp_0_dup.loc[:,'sv_type_h']="0/0"
                    df_temp_0_dup.loc[:,'sv_type_prob_h']=str(round(df_temp_00_dup.shape[0]/df_temp_0_dup.shape[0],2))
                    sv_type_h_dup="0/0"
                    sv_type_prob_h_dup=str(round(df_temp_00_dup.shape[0]/df_temp_0_dup.shape[0],2))
                
                elif ((df_temp_01_dup.shape[0]>df_temp_00_dup.shape[0]) and (df_temp_01_dup.shape[0]>df_temp_02_dup.shape[0])):
                    df_temp_0_dup.loc[:,'sv_type_h']="DUP"
                    df_temp_0_dup.loc[:,'sv_type_prob_h']=str(round(df_temp_01_dup.shape[0]/df_temp_0_dup.shape[0],2))
                    sv_type_h_dup="DUP"
                    sv_type_prob_h_dup=str(round(df_temp_01_dup.shape[0]/df_temp_0_dup.shape[0],2))

                else:
                    df_temp_0_dup.loc[:,'sv_type_h']="INV"
                    df_temp_0_dup.loc[:,'sv_type_prob_h']=str(round(df_temp_02_dup.shape[0]/df_temp_0_dup.shape[0],2))
                    sv_type_h_dup="INV"
                    sv_type_prob_h_dup=str(round(df_temp_02_dup.shape[0]/df_temp_0_dup.shape[0],2))        
        
            # subset of all with label==1:
            df_temp_1_dup=df_csv_dup[(df_csv_dup['label'] == '1') | (df_csv_dup['label'] == '2')]
            # partitions of label 1 among predictions
            df_temp_10_dup=df_temp_1_dup[df_temp_1_dup['predictions']=='0.0']
            df_temp_11_dup=df_temp_1_dup[df_temp_1_dup['predictions']=='1.0']
            df_temp_12_dup=df_temp_1_dup[df_temp_1_dup['predictions']=='2.0']
        
        
            if df_temp_0_dup.shape[0]!=0 or df_temp_1_dup.shape[0]!=0:
            # don't include if many of them are 0's.
                if df_temp_1_dup.shape[0]==0:
                    k=k+1
                    #print(filename)
                    continue;
                elif((df_temp_00_dup.shape[0]/df_temp_0_dup.shape[0] >=0.5) and (df_temp_10_dup.shape[0]/df_temp_1_dup.shape[0] >=0.5)):
                    k=k+1
                    continue
            else:
                if (df_temp_10_dup.shape[0]/df_temp_1_dup.shape[0] >=0.5):
                    cls_qual_dup=1
                    k=k+1
                    #print(filename)
                    continue
        
            #continue
            if ((df_temp_10_dup.shape[0]>df_temp_11_dup.shape[0]) and (df_temp_10_dup.shape[0]>df_temp_12_dup.shape[0])):
                # ASK THE NAMING OF THIS
                continue
            
                #df_temp_1_dup.loc[:,'sv_type']="1/1"
                #df_temp_1_dup.loc[:,'sv_type_prob']=str(round(df_temp_10_dup.shape[0]/df_temp_1_dup.shape[0],2))
                #sv_type_dup="1/1"
                #sv_type_prob_dup=str(round(df_temp_10_dup.shape[0]/df_temp_1_dup.shape[0],2))
                #print(filename)
            

            elif ((df_temp_11_dup.shape[0]>df_temp_10_dup.shape[0]) and (df_temp_11_dup.shape[0]>df_temp_12_dup.shape[0])):
                ## only longranger annotation
                #if ((str(sv_type_lr) != 'DUP\n')):
                #    #print(filename, window_bgn, sv_type_lr )
                #    continue
                df_temp_1_dup.loc[:,'sv_type']="DUP"
                df_temp_1_dup.loc[:,'sv_type_prob']=str(round(df_temp_11_dup.shape[0]/df_temp_1_dup.shape[0],2))
                sv_type_dup="DUP"
                sv_type_prob_dup=str(round(df_temp_11_dup.shape[0]/df_temp_1_dup.shape[0],2))
            else:
                ## only longranger annotation
                #if ((str(sv_type_lr) != 'INV\n')):
                #    #print(filename, window_bgn, sv_type_lr )
                #    continue
                df_temp_1_dup.loc[:,'sv_type']="INV"
                df_temp_1_dup.loc[:,'sv_type_prob']=str(round(df_temp_12_dup.shape[0]/df_temp_1_dup.shape[0],2))
                sv_type_dup="INV"
                sv_type_prob_dup=str(round(df_temp_12_dup.shape[0]/df_temp_1_dup.shape[0],2))

            df_temp_dup = pd.concat([df_temp_1_dup,df_temp_0_dup])
            df_temp_dup = df_temp_dup.sort_values(by=['name'])
            #df_temp.to_csv('df_temp.csv')
            
            #df_temp=pd.read_csv('df_temp.csv',dtype=str, sep=',')  
            probs_dup = df_temp_dup['genotype']+':'+df_temp_dup['prob_ref'] + ',' +df_temp_dup['prob_het']+ ',' +df_temp_dup['prob_alt']
            with open(args3+'test.bed', 'w') as tb_dup:
                tb_dup.write(chr_name_dup+"\t"+str(beginning_loc_dup)+"\t"+str(int(beginning_loc_dup)+1)+"\ta")
                #tb.write("chr2\t14423253\t14423254\tab")
            tb_dup.close()
        
            #out_dup = subprocess.run(['bedtools getfasta -fi '+args3+'UCSC_mm10.fa -bed '+args3+'+test.bed'], shell=True,  capture_output=True)
            out_dup = subprocess.run(['bedtools getfasta -fi '+args3+'UCSC_mm10.fa -bed '+args3+'test.bed'], shell=True,  capture_output=True)
            base_dup = out_dup.stdout.decode().split('\n')[1]
            if (round(stats.hmean([float(cls_qual_dup),1]),3) >=0.75):
                filter_score_dup = 'PASS'
            else:
                filter_score_dup = 'LOWQ'
            if(df_temp_0_dup.shape[0]==0):
                if (round(stats.hmean([float(cls_qual_dup),float(sv_type_prob_dup)]),3) >=0.75):
                    filter_score_dup = 'PASS'
                else:
                    filter_score_dup = 'LOWQ'
                
                row_dup=[chr_name_dup, beginning_loc_dup,cls_qual_dup,'.', base_dup, '<'+sv_type_dup+'>',round(stats.hmean([float(cls_qual_dup),float(sv_type_prob_dup)]),3),filter_score_dup ,'IMPRECISE;SVTYPE='+sv_type_dup+';SVEND='+str(ending_loc_dup) +
                 ';SVLEN='+str(int(svlen_dup))+';CIPOS='+str(cipos1)+','+str(cipos2)+';CIEND='+str(ciend1)+','+str(ciend2)+';CLS_QUAL='+str(cls_qual_dup)+';QUAL='+sv_type_prob_dup, 'GT:MP']
            else:
                if (round(stats.hmean([float(cls_qual_dup),float(sv_type_prob_dup),float(sv_type_prob_h_dup)]),3) >=0.75):
                    filter_score_dup = 'PASS'
                else:
                    filter_score_dup = 'LOWQ'
                row_dup=[chr_name_dup, beginning_loc_dup,cls_qual_dup,'.', base_dup, '<'+sv_type_dup+'>',round(stats.hmean([float(cls_qual_dup),float(sv_type_prob_dup),float(sv_type_prob_h_dup)]),3),filter_score_dup ,'IMPRECISE;SVTYPE='+sv_type_dup+';SVEND='+str(ending_loc_dup) +
                 ';SVLEN='+str(int(svlen_dup))+';CIPOS='+str(cipos1)+','+str(cipos2)+';CIEND='+str(ciend1)+','+str(ciend2)+';CLS_QUAL='+str(cls_qual_dup)+';QUAL='+sv_type_prob_dup, 'GT:MP']
            row_dup = row_dup+probs_dup.tolist()
            df_gvcf_dup.loc[len(df_gvcf_dup)]=row_dup
    print(k, "regions are excluded since we cannot found any duplications or inversions in those regions")

    df_gvcf_dup.POS = df_gvcf_dup.POS.astype(float)
    df_gvcf_dup = df_gvcf_dup.sort_values(['#CHROM', 'POS'])
    df_gvcf_dup.POS = df_gvcf_dup.POS.astype(int)
    df_gvcf_dup = df_gvcf_dup.reset_index(drop=True)


    i=1
    last_index_dup=len(df_gvcf_dup['POS'])
    while i<last_index_dup:
        #print(i,last_index)
        if ((df_gvcf_dup['#CHROM'][i]==df_gvcf_dup['#CHROM'][i-1]) and (int(df_gvcf_dup['POS'][i])-int(df_gvcf_dup['POS'][i-1]))<1000):
            if df_gvcf_dup['CLS_QUAL'][i]>df_gvcf_dup['CLS_QUAL'][i-1]:
                df_gvcf_dup=df_gvcf_dup.drop(df_gvcf_dup.index[[i-1]])
                i=1
                last_index_dup = len(df_gvcf_dup['POS'])
                df_gvcf_dup = df_gvcf_dup.reset_index(drop=True)
                continue
            else:
                df_gvcf_dup=df_gvcf_dup.drop(df_gvcf_dup.index[[i]])
                df_gvcf_dup = df_gvcf_dup.reset_index(drop=True)
                i=1
                last_index_dup = len(df_gvcf_dup['POS'])
                continue
        i+=1
    del df_gvcf_dup['CLS_QUAL']

    df_all = pd.concat([df_gvcf,df_gvcf_dup], axis=0, ignore_index=True, sort=False)



    # df_all=df_gvcf

    df_all = df_all.sort_values(['#CHROM', 'POS'])

    #df_all=df_all.drop(['4512-JFI-0341_BXD149'], axis=1)

    #df_all=df_all.drop(df_all.columns[10], axis=1)

    df_chr=df_all[df_all['#CHROM']==str(args1)]

    df_chr = df_chr.reset_index(drop=True)

    df_chr.to_csv(args2+args1+'_without_header.gvcf', header=True, index=False, sep='\t', mode='w')
