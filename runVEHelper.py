# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 21:01:09 2018

@author: LENOVO
"""

import sys
import os
import socket
import time
import tensorflow as tf
sys.path.append('C:\\Ronald\\AI\\MNIST')
import VEHelperTitleOnlyTeam as v

def run(host='', port = 42504):
    os.system ('title Service - PD to Team')
    with tf.device('/cpu:0'):
        vehelper = v.VEHelper()
        vehelper.load_data()
        v.vocab_size = 3000
        v.sentence_max_len = 50
        v.hp_embedding_size = 150
        v.hp_LSTM_size = 100
        vehelper.build_model(b_load_existing_model = True, 
                             #str_load_model_pathname = 'C:\Ronald\\AI\\VEHelper\\model3\\2018-08-13-6754.lstm50.model.best.91.7')
                             #str_load_model_pathname = 'C:\Ronald\\AI\\VEHelper\\model3\\2018-08-24-9845.lstm50.toteam.model.best.93.3')
                             str_load_model_pathname = 'C:\Ronald\\AI\\VEHelper\\model3\\2018-09-01.vocab3000.embed150.lstm100.toteam.notag.model.best.94.7')
        
        with socket.socket() as s:
            s.bind((host, port))
            i = 1
            while (True):
                print ('Waiting for REQUEST - {}'.format(i))
                i += 1
                s.listen()
                connection, address = s.accept()
                time.sleep(2)
                r = connection.recv(1000000)
                r = r.decode('utf-8')
                print (r)
                if len(r.split(' ',1)) < 2:
                    continue
                request_name = r.split(' ',1)[0]
                request_data = r.split(' ',1)[1]
                if request_name == 'TitleToTeam':
                    pd_titles = request_data.split('\r\n')
                    pd_titles = list(filter(None, pd_titles))
                    print ('REQUEST : {}'.format(request_name))
                    print ('CONTENTS : {}'.format(pd_titles))
                    predict_result = vehelper.predict_output_text(pd_titles, 100)
                    connection.sendall(predict_result.encode())
                
                elif request_name == 'TitleToArea':
                    connection.sendall('Not available now!'.encode())
                
                connection.close()
                print ('RESULT : {}'.format(predict_result))
            

run()