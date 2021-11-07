## pipeline for SVJAM: 

chrname=$1
#path="/home/hakan/Documents/joint_calling/"
path="./"

# cd $path/$chrname/images/

echo "change the folder names"
#  change the folder names of images, comment this if it is not necessary
## for i in ` ls -d */ | grep "chr"` ; do
##	loc=`echo $i |sed "s/_chr.\+//"`
##	loc1=`echo $i | sed "s|/||g"`
##	loc2="${loc1}_${loc1}/"
##	mv $loc $loc2
## done


echo "make a new dir for 512*512 images for CNN"

# make a new dir for 512*512 images for CNN
mkdir $path/$chrname/images_512/
rsync -a $path/$chrname/images/chr* $path/$chrname/images_512/

echo "convert the images in the folder images_512 to 512*512"
# convert the images in the folder images_512 to 512*512
python /home/hakan/Documents/joint_calling_scripts/size_converter_in_folders.py $path/$chrname/images_512/
python ./size_converter_in_folders.py $path/$chrname/images_512/

run_del_det_cluster() 
{
        cd $path/$chrname/images/
	for i in ` ls -d */ | grep "chr"` ; do 
		loc=`echo $i |sed "s/_chr.\+//"`
		echo $i
		#echo $loc
		if [ -f "${loc}_${loc}.csv" ] || [ -f "../analysis/deletion/${loc}_${loc}.csv" ];
		then
			continue
		fi
		# python /home/hakan/Documents/joint_calling_scripts/pca_crop_cluster_del_detection.py $i $loc
		python ../../crop_pca_cluster_detection.py $i $loc
	done
}

echo "run del_detection and clustering"
# run del_detection and clustering
run_del_det_cluster

cd ../..

echo "mkdir the necessary folders for analysis"
# mkdir the necessary folders for analysis
mkdir -p $path/$chrname/analysis
mkdir -p $path/$chrname/analysis/deletion
mkdir -p $path/$chrname/analysis/del
mkdir -p $path/$chrname/analysis/del/csv_log_files
mkdir -p $path/$chrname/analysis/dup_inv
mkdir -p $path/$chrname/analysis/dup_inv/csv_log_files
mkdir -p $path/$chrname/analysis/dup_inv/pred_csv;

echo "copy csv and log files to csv_log_files folders"
# copy csv and log files to csv_log_files folders
cp $path/$chrname/images/*{csv,log} $path/$chrname/analysis/del/csv_log_files
cp $path/$chrname/images/*{csv,log} $path/$chrname/analysis/dup_inv/csv_log_files
mv $path/$chrname/images/*csv $path/$chrname/analysis/deletion
mv $path/$chrname/images/*log $path/$chrname/analysis/deletion
mv $path/$chrname/images/*png $path/$chrname/analysis/deletion

echo "run CNN prediction for duplication and inversion"
# run CNN prediction for duplication and inversion
# python /home/hakan/Documents/joint_calling_scripts/cnn_predict.py $path/$chrname/images_512/ $path/$chrname/analysis/dup_inv/csv_log_files/ $path/$chrname/analysis/dup_inv/pred_csv/

python ./cnn_predict.py $path/$chrname/images_512/ $path/$chrname/analysis/dup_inv/csv_log_files/ $path/$chrname/analysis/dup_inv/pred_csv/

echo "running gvcf_maker"
# python /home/hakan/Documents/joint_calling_scripts/gvcf_maker.py $chrname $path/$chrname/ /home/hakan/Documents/joint_calling_scripts/
python ./gvcf_maker.py $chrname $path/$chrname/ ./

mv $path/$chrname/$chrname.gvcf $path/$chrname/{$chrname}_without_header.gvcf
cat header.txt $path/$chrname/{$chrname}_without_header.gvcf > $path/$chrname/$chrname.gvcf
rm $path/$chrname/{$chrname}_without_header.gvcf
