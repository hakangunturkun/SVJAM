0#!/usr/bin/python3
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import WebDriverException
import time
import sys
import re
import os
import random
import base64
import multiprocessing as mp
from multiprocessing.pool import ThreadPool, Pool
import subprocess


#print("haddeeeeeeee111")

#os.system("pkill -9 chrome")
#subprocess.run(['gnome-terminal -- ./open_loupe.sh'], shell=True)
#time.sleep(25)

#pool = mp.Pool(mp.cpu_count())

def stringToBase64(s):
    return base64.b64encode(s.encode('utf-8'))

def base64ToString(b):
    return base64.b64decode(b).decode('utf-8')

path_images = './images_new/'
with open("./chrome.list", 'r') as locs:
    xys=locs.readlines()

#path = '/home/hakan/Documents/mouse_loupe'
path ='/media/hakan/New Volume/mouse_loupe'
downloadDir=path_images

samples=[]
for filename in os.listdir(path):
    if filename.endswith('.loupe'):
        #print(filename) 
        #print(filename.split('.loupe')[0])
        samples.append(filename.split('.loupe')[0])
random.shuffle(samples)
#sleeptime=10
port=3001

#print("haddeeeeeeee")

def browser_options():

    global serverRoot
    global options
    global profile
    global browser
    global driverversion
    global browserversion
    serverRoot = "http://127.0.0.1:"+ str(port) + "/loupe/view/"

    # disable the download dialog popup box
    options=Options()
    options.headless=True
    #profile = webdriver.Chrome()
    profile = webdriver.Chrome('/usr/bin/chromedriver')
    profile.set_preference('browser.download.folderList', 2) # custom location
    profile.set_preference('browser.download.manager.showWhenStarting', False)
    profile.set_preference('browser.download.dir', downloadDir)
    profile.set_preference('browser.helperApps.neverAsk.saveToDisk', "image/png")
    browser = webdriver.Chrome(Chrome_profile=profile, options=options)
    driverversion = browser.capabilities['moz:geckodriverVersion']
    browserversion = browser.capabilities['browserVersion']
    #print("gecko driver ver: "+driverversion)
    #print("Chrome ver: "+browserversion)
    #print("port: "+str(port))

serverRoot = "http://127.0.0.1:"+ str(port) + "/loupe/view/"

# function to take care of downloading file
def enable_download_headless(browser,download_dir):
    browser.command_executor._commands["send_command"] = ("POST", '/session/$sessionId/chromium/send_command')
    params = {'cmd':'Page.setDownloadBehavior', 'params': {'behavior': 'allow', 'downloadPath': download_dir}}
    browser.execute("send_command", params)

chrome_options = Options()
chrome_options.add_argument("--headless")

samplesLoaded=[]

def multipro(xy):
#for sample in samples:
#    for xy in list(xys):
        ## print("genomic region:" + xy.rstrip())
        xy=xy.rstrip()
        xy=xy[1:]
        xy=re.sub(r',', '', xy)
        filexy=xy
        xchr=re.search('^(chr.*?:)', xy)
        ychr=re.search(';(chr.*?:)', xy)
        xy=re.sub(r'-','-'+xchr.group(1), xy,1)
        xy=re.sub(r';(chr.*-)',';'+r'\1'+ychr.group(1), xy)
        xy=re.sub(r':', '%2B', xy)
        xy=re.sub(r';', '&y=', xy)
        filexy=re.sub(r':', '_', filexy)
        filexy=re.sub(r';', '_', filexy)
        filename=filexy+"_"+sample+".png"
        #url=serverRoot + sampleKeys[sample] + '/matrix?x='+ xy
        url=serverRoot + str(stringToBase64(sample+'.loupe')).split('\'')[1] + '/matrix?x=' + xy
        ## print ("\tdownload ... "+filename)
        
        download_dir=downloadDir+filexy +"/"
        browser = webdriver.Chrome(options=chrome_options, executable_path='/usr/local/share/chromedriver')
        enable_download_headless(browser,download_dir)
        #time.sleep(10)
        browser.get(url)
        time.sleep(1)

        try:    
            elem = browser.find_element_by_id('save_image_as_png')  # find the save button for the matrix view 
            time.sleep(0.8)
            #elem.click()
            WebDriverWait(browser, 5).until(expected_conditions.element_to_be_clickable((By.ID, "save_image_as_png"))).click()
            movecommand="mv " + download_dir + "loupe-sv-barcode-matrix.png "  +  downloadDir+filexy+'/'+filename
            time.sleep(0.7)
            os.system(movecommand)
        except TimeoutException:
            print('Time Out Error')
            browser.quit()
            time.sleep(0.1)
        browser.quit()

filexy_ls=[]
temp_flag=0
for xy in list(xys):
    xy=xy.rstrip()
    xy=xy[1:]
    y=re.sub(r',', '', xy)
    filexy=xy
    xchr=re.search('^(chr.*?:)', xy)
    ychr=re.search(';(chr.*?:)', xy)
    xy=re.sub(r'-','-'+xchr.group(1), xy,1)
    xy=re.sub(r';(chr.*-)',';'+r'\1'+ychr.group(1), xy)
    xy=re.sub(r':', '%2B', xy)
    xy=re.sub(r';', '&y=', xy)
    filexy=re.sub(r':', '_', filexy)
    filexy=re.sub(r';', '_', filexy)
    if not os.path.exists(downloadDir+filexy):
        os.makedirs(downloadDir+filexy)
        filexy_ls.append(downloadDir+filexy)
    else:
        if (temp_flag==0):
            for name in os.listdir(downloadDir):
                temp_flag=1
                filexy_ls.append(name)
        #print([name for name in os.listdir('./images/') if os.path.isdir(name)])

chrome_options = Options()
chrome_options.add_argument("--headless")        

for sample in samples:
    start=time.time()
    #os.system("pkill -9 chrome")
    for files in filexy_ls:
        #print(files)
        path=downloadDir+files+"/"+files+"_"+sample+".png"
        if  os.path.exists(path) == False:
            try:
                #os.system("pkill -9 chrome")
                pool = Pool(processes=1)
                map_results_list = pool.map(multipro, xys)
            except WebDriverException:
                #subprocess.run(['gnome-terminal pkill chrome'], shell=True)
                print("WebDriverException is here!")
                #os.system("pkill -9 chrome")
                #sys.exit()
                #subprocess.run(['gnome-terminal -- ./open_loupe.sh'], shell=True)
                #time.sleep(25)
                #pool = Pool(processes=25)
                map_results_list = pool.map(multipro, xys)
    end=time.time()
    print(sample, ":" , end-start)


#browser.quit()

