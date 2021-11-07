# give the loc of the folders
folders_loc=$1

# give the name of the original folder
orig_folder=$2

# give the name of the new folder
new_folder=$3

cd $folders_loc

mkdir $folders_loc/$new_folder/
mkdir $folders_loc/$new_folder/images
# copy the original folder under a new name
rsync -a $folders_loc/$orig_folder/* $folders_loc/$new_folder/images

#mkdir $folders_loc/$new_folder/images
#mv $folders_loc/$orig_folder/* $folders_loc/$new_folder/images

cd $folders_loc/$new_folder/images
echo "small sized:"
find . -name "*.png" -type 'f' -size -1k
find . -name "*.png" -type 'f' -size -1k -delete
echo "lanes":
find . -name "*lanes*" -type 'f'
find . -name "*lanes*" -type 'f' -delete
echo "gene:"
find . -name "*gene*" -type 'f'
find . -name "*gene*" -type 'f' -delete
echo "JFI:"
find . ! -name "*JFI*" -type 'f'
find . ! -name "*JFI*" -type 'f' -delete

find . -maxdepth 1 -type d -exec bash -c "echo -ne '{} '; ls '{}' | wc -l" \; |   awk '$NF!=152'
