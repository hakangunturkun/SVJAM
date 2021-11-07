# SVJAM: structural variant joint analysis by machine learning

SVJAM jointly detects and genotypes large structural variants (SVs) from linked-read whole genome sequence data.

## Citation:

Gunturkun, M. H., Villani, F., Colonna, V., Ashbrook, D., Williams, R. W.,and Chen, H. (2021). Svjam:  Joint analysis of structural variants usinglinked read sequencing data.bioRxiv. doi:10.1101/2021.11.02.467006

## Flowchart:

![Alt text](../flowchart/pipeline.png?raw=true "Summary of the pipeline")

## Running:

```
./SVJAM chr*
```

## Test example:

* Test SVJAM by running: 
```
chmod +x SVJAM
./SVJAM chr1
```

* We provide an example image set for testing in **Example/chr1/images/** folder. 

* The required folder for test is **Example/chr1/images** folder with image files in it. Image files are organized in folders named as: *chr1_93011035-93221780_chr1_93011035-93221780*

* The output will be formed in **Example/chr1/** folder. You should see the output as in **Example/chr1/output** folder.

## Downloading images:
* Install loupe server from 10X Genomics.
* Generate *.loupe* files by using linked-read WGS.
* Open and organize **image_download/open_loupe.sh** then run it.
* Download images manually or use **image_download/chrome_multi_save_matrixView_pngs.py** for automatically download that uses Selenium library.

## Dependencies

* python == 3.8
* see requirements.md for list of packages and versions
