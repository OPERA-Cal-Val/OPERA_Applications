from lxml import etree
import requests

# Function to get cloud coverage from provisional product XMLs on S3
def get_cc(xmlpath:str):
    '''
    Returns percent cloud cover for each date.
            Parameters:
                    xmlpath (str): XML file for each date
            Returns:
                    cc (int): percent cloud cover for each date
    '''
    key = xmlpath
    url = f'https://opera-pst-rs-pop1.s3.us-west-2.amazonaws.com/{key}'
    metadata_tree = etree.fromstring(requests.get(url).content)
    atts = metadata_tree.find('.//{http://www.isotc211.org/2005/gmd}contentInfo/{http://www.isotc211.org/2005/gmd}MD_CoverageDescription/{http://www.isotc211.org/2005/gmd}dimension/{http://www.isotc211.org/2005/gmd}MD_Band/{http://www.isotc211.org/2005/gmd}otherProperty/{http://www.isotc211.org/2005/gco}Record/{http://earthdata.nasa.gov/schema/eos}AdditionalAttributes')
    for elem in atts:
        att_name = elem.find('.//{http://earthdata.nasa.gov/schema/eos}EOS_AdditionalAttributeDescription').find('./{http://earthdata.nasa.gov/schema/eos}name').find('./{http://www.isotc211.org/2005/gco}CharacterString').text
        if att_name == 'PercentCloudCover':
            cc = int(elem.find('./{http://earthdata.nasa.gov/schema/eos}value').find('{http://www.isotc211.org/2005/gco}CharacterString').text)
    return cc
