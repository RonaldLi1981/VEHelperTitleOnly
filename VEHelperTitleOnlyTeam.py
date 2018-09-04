# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 22:46:05 2018

@author: Ronald Li
"""
import os
import threading
import tkinter as tk
import csv
import datetime
import tensorflow as tf
from tensorflow import keras as k #, Bidirectional, LSTM, Dropout, Dense
from tool_mib_word_split import sentenceWordSplit
from setting import backend_work_path

# MONITOR WINDOW
root_wnd = tk.Tk()
root_wnd.title('Training Monitor')
root_wnd.geometry('400x300')
root_wnd.resizable(width=0, height=0)
root_wnd.attributes('-topmost', True)
font_big = 'Arial 25 bold'
font_middle = 'Arial 15'
font_small = 'Consolas 11'
dt_start = datetime.datetime.now()
dt_delta = datetime.timedelta()
lb_timer = tk.Label(root_wnd, text = '00:00:00', font = font_middle)
lb_timer.pack()
lb1 = tk.Label(root_wnd, text = 'BEST : ', font = font_middle)
lb1.pack()
lb2 = tk.Label(root_wnd, text = 'N/A', font = font_big)
lb2.pack()
lb3 = tk.Label(root_wnd, text = 'CURRENT : ', font = font_middle)
lb3.pack()
lb4 = tk.Label(root_wnd, text = 'N/A', font = font_big)
lb4.pack()
tx1 = tk.Text(root_wnd, height = 8, width = 49, font = font_small)
tx1.insert('end', '    Training Not Started Yet...')
tx1.pack()

# GENERAL PARAM
WORK_PATH = backend_work_path

# MODEL PARAMETER
vocab_size = 3000 # originalvalue = 3000
sentence_max_len = 50 # originalvalue = 50
dictArea = {'QA2':0, 'QA3&7':1, 'QA4':2, 'QA6':3, 'BLR':4}
dictAreaInvert = {0:'QA2', 1:'QA3&7', 2:'QA4', 3:'QA6', 4:'BLR'}
hp_categories = 5
hp_embedding_size = 100 # originalvalue = 256
hp_LSTM_size = 200
hp_dropout = (1-0.2)
#hp_dropout2 = 0.0


# TRAINING PARAMETER
# training_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.AllAreas.Clear.S.Corrected.csv'
training_filepathname = WORK_PATH + 'Data\\Training.9881.ToTeam.NoTag.RANDOM.csv'
# training_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.AllAreasClear.FinalTest.RANDOM.csv'
training_data_qty = 9892 # USELESS NOW
quick_training_rounds = 0
training_rounds = 500 # total training epochs = training_times * training_epochs
training_epochs = 1
batch_size = 500
save_model_filepathname = WORK_PATH + 'model3\\{}.vocab{}.embed{}.lstm{}.toteam.notag.model'.format(datetime.datetime.now().strftime('%Y-%m-%d'), vocab_size, hp_embedding_size, hp_LSTM_size)
save_bestmodel_filepathname =  save_model_filepathname + '.best'
training_log_filepathname = WORK_PATH + 'TrainingLog\\VE Helper - Training Log - 9845 - lstm{} - toteam.csv'.format(hp_LSTM_size)
verbose_mode = 2

# TEST PARAMETER
final_test_filepathname = WORK_PATH + 'Data\\FinalTest2.ToTeam.NoTag.csv'
# final_test_filepathname = 'C:\Ronald\\AI\\VEHelper\\Data\\AI.PD.FromFeatureContent.AR0102011.85.csv'
predict_output_filepathname = WORK_PATH + 'Data\\Predict_Output.FinalTest2.ToTeam.NoTag.csv'
SWITCH_PRINT_FINALTEST_DETAILS = False

# EXECUTION PARAMETER
SWITCH_TRAIN = True  #是否要训练
SWITCH_EVALUATE = False     #训练中是否执行评估环节（基本上现在已经被废弃不用，因为我更喜欢用FinalTest去验证）
SWITCH_LOAD_EXISTING_MODEL = False  #加载现存模型参数。模型的文件名由下一行这个参数给出。
model_to_load_pathname = save_model_filepathname #'C:\Ronald\\AI\\VEHelper\\model3\\2018-08-24-9845.lstm50.toteam.model.best.95.3'
SWITCH_FINAL_TEST = True

# OTHER PARAMETER
best_model_score = 0
best_model_score_round_number = 0
graph = tf.get_default_graph()

class VEHelper:
    def __init__ (self):
        self.tknz = k.preprocessing.text.Tokenizer(num_words = vocab_size)
        self.model = None
        self.sqPdTexts_train = []
        self.sqPdTexts_test = []
        self.catLabels_train = []
        self.catLabels_test = []
        self.sentence_length = sentence_max_len
        self.dictAreaInvert = dictAreaInvert.copy()
    
    def load_data (self):
        f = open(training_filepathname, 'r', encoding='iso-8859-15')
        lines = f.readlines()
        fields = []
        PdTexts = []
        labels = []
        for line in lines:
            fields = line.split(sep=',',maxsplit=1)
            labels.append(dictArea[fields[0]])
            PdTexts.append(sentenceWordSplit(fields[1]))
        print ('\n\n=== Training Data Loading Completed =====\n')
        f.close()
    
        self.tknz.fit_on_texts (PdTexts)
        sqPdTexts = self.tknz.texts_to_sequences(PdTexts)
        sqPdTexts = k.preprocessing.sequence.pad_sequences(sqPdTexts, maxlen=sentence_max_len)
        self.sqPdTexts_train = sqPdTexts.copy()
        if (SWITCH_EVALUATE):
            self.sqPdTexts_test = sqPdTexts[training_data_qty:]
        
        catLabels = k.utils.to_categorical(labels, hp_categories)
        self.catLabels_train = catLabels.copy()
        if (SWITCH_EVALUATE): 
            self.catLabels_test = catLabels[training_data_qty:]

    # Model
    def build_model(self, b_load_existing_model = False, str_load_model_pathname = ''):
        print ('building model ...')
        self.model = k.Sequential()
        self.model.add(k.layers.Embedding(vocab_size, hp_embedding_size, input_length=sentence_max_len, mask_zero=True))
        self.model.add(k.layers.Bidirectional(k.layers.LSTM(hp_LSTM_size,implementation=1,dropout=hp_dropout,activation='tanh')))
        #self.model.add(k.layers.Bidirectional(k.layers.LSTM(128,implementation=2,dropout=0.3,activation='sigmoid')))
#        self.model.add(k.layers.Dropout(hp_dropout2))
        self.model.add(k.layers.Dense(hp_categories, activation='softmax'))
        self.model.compile('RMSprop', 'categorical_crossentropy', metrics=['accuracy'])
        #加载上一个训练好的模型
        if (b_load_existing_model):
            if (len(str_load_model_pathname) > 0):
                self.model.load_weights(str_load_model_pathname)
            else:
                self.model.load_weights(save_model_filepathname)

    def train(self,training_epochs=20):
        print ('training model ...')
        self.model.fit(self.sqPdTexts_train, self.catLabels_train,verbose=verbose_mode,batch_size=batch_size,epochs=training_epochs)
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
            PdTexts.append(sentenceWordSplit(fields[1]))
            answers.append(fields[0])
            line_counter += 1
        sqPdTexts = self.tknz.texts_to_sequences(PdTexts)
        sqPdTexts = k.preprocessing.sequence.pad_sequences(sqPdTexts, maxlen=sentence_max_len)
        model_predicts = self.model.predict_classes(sqPdTexts, batch_size=100)

        correct_counter = 0
        correct_counter_area = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        incorrect_counter_area = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        writer_out.writerow (['Correct?','''Model's Answer''','Standard Answer','PD Title'])
        for i in range(0, line_counter):
            str_correct_or_not = 'NO'
            if (dictAreaInvert[model_predicts[i]] == answers[i]):
                str_correct_or_not = 'YES'
                correct_counter += 1
                correct_counter_area[dictArea[answers[i]]] += 1
            else:
                incorrect_counter_area[dictArea[answers[i]]] += 1
            writer_out.writerow ([str_correct_or_not, dictAreaInvert[model_predicts[i]] , answers[i] , PdTexts[i]])
            #print ('Model Predict: ' + dictAreaInvert[answers[0]] + '  FOR  ' +  sqPdText)
            if (SWITCH_PRINT_FINALTEST_DETAILS):
                print ('Model Predict: ' + dictAreaInvert[model_predicts[i]] + '  FOR  ' + answers[i] + '  '+  PdTexts[i])
        fout.close()
        
        f_training_log = open(training_log_filepathname, 'a', encoding='iso-8859-15', newline = '')
        str_final_test_gui_text = ''
        str_final_test_gui_text += '      {} / {} ( {:2.3f} % )\n'.format( correct_counter, line_counter, correct_counter*100/line_counter)
        print ('\n    {} inputs tested, {} answers are correct ( {}% )'.format(line_counter, correct_counter, correct_counter*100/line_counter))
        writer_training_log = csv.writer(f_training_log)
        print ('    Failures Per Area')
        str_failure_per_area = ''
        for i in range(0,hp_categories):
            i_a = incorrect_counter_area[i]
            i_b = correct_counter_area[i]
            if (i_a + i_b > 0):
                i_c = i_a * 100 / (i_a + i_b)
            else:
                i_c = 0
            print ('\t\t{:>16} : {:>3} / {:<3}\t- {:2.0f}%'.format(dictAreaInvert[i],i_a, i_a+i_b, i_c ))
            str_failure_per_area += '{:>18} : {:>3} / {:<3} - {:2.0f}%\n'.format(dictAreaInvert[i],i_a, i_a+i_b, i_c )
        str_final_test_gui_text += str_failure_per_area
        global tx1
        tx1.delete(1.0, 'end')
        tx1.insert('end', str_final_test_gui_text)
        root_wnd.update()
            
        writer_training_log.writerow ([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),'{} epochs trained'.format(training_epochs),'{} inputs tested, {} answers are correct ( {}% )'.format(line_counter, correct_counter, correct_counter*100/line_counter),str_failure_per_area])
        f_training_log.close()

        #print ('=== Final Test Done =====')
        return correct_counter*100/line_counter

    def predict_output_text(self, pd_titles, batch_size=1):
        pd_titles_ws = []
        for i in range(0, len(pd_titles)):
            pd_titles_ws.append ( sentenceWordSplit(pd_titles[i]) )
        sqQuestion = self.tknz.texts_to_sequences(pd_titles_ws)
        sqPdQuestion = k.preprocessing.sequence.pad_sequences(sqQuestion, maxlen=self.sentence_length)
        results = self.model.predict_classes(sqPdQuestion, batch_size)
        results_text_with_title = ''
        for i in range(0, len(results)):
            if (len(pd_titles[i].split()) < 4):
                results_text_with_title += (pd_titles[i] + '\r\n')
            else:
                results_text_with_title += (dictAreaInvert[results[i]] + '\t' + pd_titles[i] + '\r\n')
        return results_text_with_title

def main():
    # 生成主窗体
    global root_wnd, lb1, lb2, lb3, lb4
    global best_model_score, best_model_score_round_number
    root_wnd.update()
    thread1 = threading.Thread(target=workThreadFunction)
    thread1.start()
    root_wnd.mainloop()
    
def workThreadFunction ():         
    global root_wnd, dt_delta, lb_timer, lb1, lb2, lb3, lb4, graph
    global best_model_score, best_model_score_round_number
    with tf.device('/gpu:0'):
        graph.as_default()
        vehelper = VEHelper()
        vehelper.load_data()
        vehelper.build_model(SWITCH_LOAD_EXISTING_MODEL, model_to_load_pathname)
        best_model_score = vehelper.run_final_test()
        # 先进行N轮的快速训练，中间不要停下来做FinalTest，提高运行效率
        if (SWITCH_TRAIN):
            if (quick_training_rounds != 0):
                print ('[ PRE ]  === Start {} Round of Quick Training\n'.format(quick_training_rounds))
                vehelper.train(training_epochs * quick_training_rounds)
        for i in range(0+quick_training_rounds, training_rounds+quick_training_rounds):
            if (SWITCH_TRAIN):
                print ('[ {} ]  === Start The {} / {} Round of Training\n'.format(i+1,i+1,training_rounds))
                vehelper.train(training_epochs)
            if (SWITCH_FINAL_TEST):
                current_score = vehelper.run_final_test()
                lb4['text'] = '{:2.1f} @ {} ROUND'.format(current_score, 1+i)
                root_wnd.update()
                if (current_score > best_model_score):
                    if (os.path.exists(save_bestmodel_filepathname + '.{:2.1f}'.format(best_model_score))):
                        os.remove(save_bestmodel_filepathname + '.{:2.1f}'.format(best_model_score))
                    vehelper.model.save(save_bestmodel_filepathname + '.{:2.1f}'.format(current_score))
                    best_model_score = current_score
                    best_model_score_round_number = 1+i
                    lb2['text'] = '{:2.1f} @ {} ROUND'.format(best_model_score, best_model_score_round_number)
                    root_wnd.update()
                print ('    Best Score is [ {:2.1f} ], got in [ Round {} ]\n'.format(best_model_score, best_model_score_round_number))
            dt_delta = datetime.datetime.now() - dt_start
            lb_timer['text'] = (datetime.datetime(2000, 1, 1, 0, 0, 0) + dt_delta).strftime('%H:%M:%S')
            root_wnd.update()
    
        # 这段是单条PD标题的测试    
        if (True):
            question = ['SERVICE ELECTRIC - CW/3WC not working consistently (TM822/TS9.1.103S5P.SIP)']
            sqQuestion = vehelper.tknz.texts_to_sequences(question)
            print (sqQuestion)
            sqPdQuestion = k.preprocessing.sequence.pad_sequences(sqQuestion, maxlen=sentence_max_len)
            print(sqPdQuestion)
            predict_area = vehelper.model.predict_classes(sqPdQuestion, batch_size=1)
            print(dictAreaInvert[predict_area[0]])
    

if __name__=="__main__":
    main()