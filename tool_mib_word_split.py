# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:02:36 2018

@author: Ronald Li
"""

def mibWordSplit(mib):
    if mib == 'WiFi':
        return mib
    
    capital_char_index = [0]
    for i in range(1, len(mib)):
        if (mib[i].isupper() or mib[i].isdigit()):
            if (capital_char_index[-1] +1 != i):
                capital_char_index.append (i)

    seperated_words = ''
    for i in range(1, len(capital_char_index)):
        seperated_words += mib[capital_char_index[i-1]:capital_char_index[i]] + ' '
    
    seperated_words = seperated_words[0:(len(seperated_words)-1)]
    
    if seperated_words == '':
        seperated_words = mib
    return seperated_words
    

def sentenceWordSplit(sentence, mib_length_threshold = 12):
    if (sentence == ''):
        return ''
    
    sentence = sentence.replace('Wi-Fi', 'WiFi')
    sentence = sentence.replace(':', ' ')
    sentence = sentence.replace(',', ' ')
    sentence = sentence.replace('(', ' ')
    sentence = sentence.replace(')', ' ')
    sentence = sentence.replace('_', ' ')
    sentence = sentence.replace('-', ' ')
    sentence = sentence.replace('"', ' ')
    sentence = sentence.replace('\'', ' ')
    sentence = sentence.replace('/', ' ')
    sentence = sentence.replace('\\', ' ')
    sentence = sentence.replace('[', ' ')
    sentence = sentence.replace(']', ' ')
    words = sentence.split()
    seperated_sentence = ''
    for word in words:
        if (len(word) >= mib_length_threshold):
            word = mibWordSplit(word)
        seperated_sentence += (word + ' ')
    seperated_sentence = seperated_sentence[0:(len(seperated_sentence)-1)]
    return seperated_sentence

#

#a =  'ietfpktcsig/dev/cidmode' #'ietfPktcSigDevCidMode'
#s1 = '[MTA MIB] The default value of MIB ietfPktcSigDevCidMode is dtAsETS'
#s2 = '''[RDKB][Puma7][MoCA]MoCA privacy can't be enabled/disabled and MoCA password can't be configured via GUI'''
#s3 = '[TG3452/DG3450] Mobile Application sdk (LCA) : Managed Device is not getting enabled when parental control is enabled from Mobile Application.' #'[Puma7] Wrong value of Device.RouterAdvertisement.InterfaceSetting.1.Interface'
#b = mibWordSplit(a)
#ss = sentenceWordSplit(s3)
#
#print (b)
#print (ss)



