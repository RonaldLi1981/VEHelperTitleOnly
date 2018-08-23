# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 20:46:00 2018

@author: Ronald Li
"""
import os
import socket
import csv
import datetime
import tensorflow as tf
from tensorflow import keras as k #, Bidirectional, LSTM, Dropout, Dense

# MODEL PARAMETER
vocab_size = 3000
sentence_max_len = 50 # originalvalue = 50
dictArea = {'Cable Modem':0, 'DOCSIS':0, 'MTA':1, 'Gateway':2, 'WiFi':3, 'GUI':4, 'TR-69':5, 'Security':6, 'MoCA': 7, 'Throughput': 8,'PacketAce': 9, 'Manufacturing':10, 'Documentation': 10, 'Other': 10}
dictAreaInvert = {0: 'Cable Modem', 1: 'MTA', 2: 'Gateway', 3: 'WiFi', 4: 'GUI', 5: 'TR-69', 6: 'Security', 7: 'MoCA', 8: 'Throughput', 9: 'PacketAce', 10 : 'Other'}
hp_embedding_size = 256
hp_LSTM_size = 50
hp_dropout = 0.3
hp_dropout2 = 0.7

# TRAINING PARAMETER
# training_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.AllAreas.Clear.S.Corrected.csv'
training_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.AllAreas.Clear.S.Corrected.6754.EnhancedMoCA.csv'
# training_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.AllAreasClear.FinalTest.RANDOM.csv'
training_data_qty = 9845
training_times = 40 # total training epochs = training_times * training_epochs
training_epochs = 25
batch_size = 50
save_model_filepathname = 'C:\Ronald\\AI\\VEHelper\\model3\\2018-08-13-6754.lstm50.model'
save_bestmodel_filepathname =  save_model_filepathname + '.best'
training_log_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\VE Helper - Training Log - 6754 - lstm50.csv'

# TEST PARAMETER
final_test_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.AllAreasClear.FinalTest2.RANDOM.csv'
# final_test_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.FromFeatureContent.AR0102011.85.csv'
predict_output_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\Predict_Output.FinalTest2.csv'
SWITCH_PRINT_FINALTEST_DETAILS = False

# EXECUTION PARAMETER
SWITCH_TRAIN = True
SWITCH_EVALUATE = False
SWITCH_LOAD_LAST_MODEL = True
SWITCH_FINAL_TEST = True

# OTHER PARAMETER
best_model_score = 0

class VEHelper:
    def __init__ (self):
        self.tknz = k.preprocessing.text.Tokenizer(num_words = vocab_size)
        self.model = None
        self.sqPdTexts_train = []
        self.sqPdTexts_test = []
        self.catLabels_train = []
        self.catLabels_test = []
    
    def load_data (self):
        f = open(training_filepathname, 'r', encoding='iso-8859-15')
        lines = f.readlines()
        fields = []
        PdTexts = []
        labels = []
        for line in lines:
            fields = line.split(sep=',',maxsplit=1)
            labels.append(dictArea[fields[0]])
            PdTexts.append(fields[1])
        print ('\n\n=== Training Data Loading Completed =====\n')
        f.close()
    
        self.tknz.fit_on_texts (PdTexts)
        sqPdTexts = self.tknz.texts_to_sequences(PdTexts)
        sqPdTexts = k.preprocessing.sequence.pad_sequences(sqPdTexts, maxlen=sentence_max_len)
        self.sqPdTexts_train = sqPdTexts[:training_data_qty]
        if (SWITCH_EVALUATE):
            self.sqPdTexts_test = sqPdTexts[training_data_qty:]
        
        catLabels = k.utils.to_categorical(labels, 9)
        self.catLabels_train = catLabels[:training_data_qty]
        if (SWITCH_EVALUATE):
            self.catLabels_test = catLabels[training_data_qty:]


#for catLabel in catLabels:
#    print (catLabel)


    def build_model(self):
        print ('building model ...')
        self.model = k.Sequential()
        self.model.add(k.layers.Embedding(vocab_size, hp_embedding_size, input_length=sentence_max_len))
        self.model.add(k.layers.Bidirectional(k.layers.LSTM(hp_LSTM_size,implementation=1,dropout=hp_dropout,activation='sigmoid')))
        #self.model.add(k.layers.Bidirectional(k.layers.LSTM(128,implementation=2,dropout=0.3,activation='sigmoid')))
        self.model.add(k.layers.Dropout(hp_dropout2))
        self.model.add(k.layers.Dense(9, activation='softmax'))
        self.model.compile('RMSprop', 'categorical_crossentropy', metrics=['accuracy'])
        #加载上一个训练好的模型，就保留下面这行
        if (SWITCH_LOAD_LAST_MODEL):
            self.model.load_weights(save_model_filepathname)

    def train(self,training_epochs=20):
        
        print ('training model ...')
        self.model.fit(self.sqPdTexts_train, self.catLabels_train,verbose=1,batch_size=batch_size,epochs=training_epochs)
        self.model.save(save_model_filepathname)
        if (SWITCH_EVALUATE):
            evaluate_score = self.model.evaluate(self.sqPdTexts_test,self.catLabels_test)
            print (evaluate_score)

    def run_final_test(self):
        f = open(final_test_filepathname, 'r', encoding='iso-8859-15')
        lines = f.readlines()
        csv.writer(f)
        fields = []
        PdTexts = []
        model_predicts = []
        fout = open(predict_output_filepathname, 'w', encoding='iso-8859-15', newline = '')
        writer_out = csv.writer(fout)
        answers = []
        line_counter = 0
        for line in lines:
            fields = line.split(sep=',',maxsplit=1)
            PdTexts.append(fields[1])
            answers.append(fields[0])
            line_counter += 1
        sqPdTexts = self.tknz.texts_to_sequences(PdTexts)
        sqPdTexts = k.preprocessing.sequence.pad_sequences(sqPdTexts, maxlen=sentence_max_len)
        model_predicts = self.model.predict_classes(sqPdTexts, batch_size=1)

        correct_counter = 0
        correct_counter_area = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        incorrect_counter_area = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(0, line_counter):
            writer_out.writerow ([dictAreaInvert[model_predicts[i]] , answers[i] , PdTexts[i]])
            if (dictAreaInvert[model_predicts[i]] == answers[i]):
                correct_counter += 1
                correct_counter_area[dictArea[answers[i]]] += 1
            else:
                incorrect_counter_area[dictArea[answers[i]]] += 1
            #print ('Model Predict: ' + dictAreaInvert[answers[0]] + '  FOR  ' +  sqPdText)
            if (SWITCH_PRINT_FINALTEST_DETAILS):
                print ('Model Predict: ' + dictAreaInvert[model_predicts[i]] + '  FOR  ' + answers[i] + '  '+  PdTexts[i])
        fout.close()
        
        f_training_log = open(training_log_filepathname, 'a', encoding='iso-8859-15', newline = '')
        print ('\n    {} inputs tested, {} answers are correct ( {}% )'.format(line_counter, correct_counter, correct_counter*100/line_counter))
        writer_training_log = csv.writer(f_training_log)
        print ('    Failures Per Area')
        str_failure_per_area = ''
        for i in range(0,11):
            i_a = incorrect_counter_area[i]
            i_b = correct_counter_area[i]
            if (i_a + i_b > 0):
                i_c = i_a * 100 / (i_a + i_b)
            else:
                i_c = 0
            print ('\t\t{:>16} : {:>3} / {:<3}\t- {:2.0f}%'.format(dictAreaInvert[i],i_a, i_a+i_b, i_c ))
            str_failure_per_area += '\t\t{:>16} : {:>3} / {:<3}\t- {:2.0f}%\n'.format(dictAreaInvert[i],i_a, i_a+i_b, i_c )
            
        writer_training_log.writerow ([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'{} epochs trained'.format(training_epochs),'{} inputs tested, {} answers are correct ( {}% )'.format(line_counter, correct_counter, correct_counter*100/line_counter),str_failure_per_area])
        f_training_log.close()

        print ('\n\n=== Final Test Done =====\n')
        return correct_counter*100/line_counter

def main():
    vehelper = VEHelper()
    vehelper.load_data()
    vehelper.build_model()
    best_model_score = vehelper.run_final_test()
    for i in range(0, training_times):
        if (SWITCH_TRAIN):
            print ('=== Start The {} / {} Round of Training\n'.format(i+1,training_times))
            vehelper.train(training_epochs)
        if (SWITCH_FINAL_TEST):
            current_score = vehelper.run_final_test()
            if (current_score > best_model_score):
                if (os.path.exists(save_bestmodel_filepathname + '.{:2.1f}'.format(best_model_score))):
                    os.remove(save_bestmodel_filepathname + '.{:2.1f}'.format(best_model_score))
                vehelper.model.save(save_bestmodel_filepathname + '.{:2.1f}'.format(current_score))
                best_model_score = current_score

    # 这段是单条PD标题的测试    
    if (True):
        question = ['SERVICE ELECTRIC - CW/3WC not working consistently (TM822/TS9.1.103S5P.SIP)']
        sqQuestion = vehelper.tknz.texts_to_sequences(question)
        print (sqQuestion)
        sqPdQuestion = k.preprocessing.sequence.pad_sequences(sqQuestion, maxlen=sentence_max_len)
        print(sqPdQuestion)
        predict_area = vehelper.model.predict_classes(sqPdQuestion, batch_size=1)
        print(dictAreaInvert[predict_area[0]])


def run(host='', port = 42504):
    vehelper = VEHelper()
    vehelper.load_data()
    global SWITCH_LOAD_LAST_MODEL
    temp = SWITCH_LOAD_LAST_MODEL
    SWITCH_LOAD_LAST_MODEL = True
    vehelper.build_model()
    SWITCH_LOAD_LAST_MODEL = temp
    
    with socket.socket() as s:
        s.bind((host, port))
        while (True):
            s.listen()
            connection, address = s.accept()
            r = connection.recv(1000)
            r = r.decode('utf-8')
            if len(r.split()) < 2:
                continue
            request = r.split()[0]
            contents = r.split()[1]
            print ('REQUEST : {}'.format(request))
            print ('CONTENTS : {}'.format(contents))
            connection.sendall('HERE YOU GO!')
            
if __name__=="__main__":
    main()