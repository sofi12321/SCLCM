# Load Chapman (12 lead denoised) data
!wget https://figshare.com/ndownloader/files/15652862

# Save here
!mkdir /content/SecondaryHDD
!mkdir /content/SecondaryHDD/chapman_ecg

# Unzip
import zipfile
with zipfile.ZipFile("/content/15652862", 'r') as zip_ref:
    zip_ref.extractall("/content/SecondaryHDD/chapman_ecg")

import shutil

# Load files with info 
!wget -q https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653762/AttributesDictionary.xlsx
shutil.move("/content/AttributesDictionary.xlsx", "/content/SecondaryHDD/chapman_ecg/AttributesDictionary.xlsx")

!wget -q https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15653771/Diagnostics.xlsx
shutil.move("/content/Diagnostics.xlsx", "/content/SecondaryHDD/chapman_ecg/Diagnostics.xlsx")

!wget -q https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651293/ConditionNames.xlsx
shutil.move("/content/ConditionNames.xlsx", "/content/SecondaryHDD/chapman_ecg/ConditionNames.xlsx")

!wget -q https://s3-eu-west-1.amazonaws.com/pfigshare-u-files/15651296/RhythmNames.xlsx
shutil.move("/content/RhythmNames.xlsx", "/content/SecondaryHDD/chapman_ecg/RhythmNames.xlsx")

# Load modification code
!git clone https://github.com/danikiyasseh/loading-physiological-data

# Change code to satisfy current info (filepath, grouptype)
import re
def modify(filepath, from_, to_):
    file = open(filepath,"r+")
    text = file.read()
    pattern = from_
    splitted_text = re.split(pattern,text)
    modified_text = to_.join(splitted_text)
    with open(filepath, 'w') as file:
        file.write(modified_text)

filepath = "/content/loading-physiological-data/load_chapman_ecg.py"

pattern = "trial = \'.*\'"
to_ = "trial = 'contrastive_msml'"
modify(filepath, pattern , to_)

pattern = "basepath = \'.*\'"
to_ = "basepath = '/content/SecondaryHDD/chapman_ecg'"
modify(filepath, pattern , to_)

pattern = "database = pd\.read\_csv\(os\.path\.join\(basepath,\'Diagnostics\.csv\'\)\)"
to_ = "database = pd.read_excel(os.path.join(basepath,'Diagnostics.xlsx'))"
modify(filepath, pattern , to_)

pattern = "database_with_dates = pd\.concat\(\(database,dates\),1\)"
to_ = "database_with_dates = pd.concat((database,dates),axis=1)"
modify(filepath, pattern , to_)

# Modify dataset to raw splitted data
!python /content/loading-physiological-data/load_chapman_ecg.py
