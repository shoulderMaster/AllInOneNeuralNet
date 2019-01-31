#!/usr/bin/python3
from config import *
import tensorflow as tf
import pandas as pd
import os
import sys
from random import shuffle
from shutil import copy2
from math import sqrt


class Model :
    def __init__(self,NN, LSTM=None, DNN=None) :

        if LSTM == True :
            self.train_op, self.loss_op, self.Y_pred_op, self.saver, self.X, self.Y, self.keep_prob, self.optimizer\
            = NN._makeSimpleLSTMGraph()

        if DNN == True :
            self.train_op, self.loss_op, self.Y_pred_op, self.saver, self.X, self.Y, self.keep_prob, self.optimizer\
            = NN._makeMultipleIndependentDNNGraph()

class InputData : 
    
    def __init__(self, inputDataConfiguation, LSTM=None, DNN=None) :

        if LSTM == True :
            self.train_x, self.train_y, self.test_x, self.test_y, self.testSetIndex\
            = self.getInputData(\
                    pathOfDNNinputData=inputDataConfiguation.pathOfinputData,\
                    train_ratio=inputDataConfiguation.train_ratio,\
                    num_input=inputDataConfiguation.num_input,\
                    num_recurrence=inputDataConfiguation.numRecurrent,\
                    normalization_method="minmax",\
                    data_randomization=True,\
                    data_uniformalization=False,\
                    RNN=True)

        if DNN == True :
            self.train_x, self.train_y, self.test_x, self.test_y, self.testSetIndex\
            = self.getInputData(\
                    pathOfDNNinputData=inputDataConfiguation.pathOfinputData,\
                    train_ratio=inputDataConfiguation.train_ratio,\
                    num_input=inputDataConfiguation.num_input,\
                    normalization_method="minmax",\
                    data_randomization=True,\
                    data_uniformalization=False,\
                    RNN=False)
        
        self.originDf = pd.read_csv("RNN_input_data_withDeratedPower.csv").set_index("datetime")
    

    def _splitXY(self, df, num_input) :
        x_df = df.iloc[:,:num_input]
        y_df = df.iloc[:,num_input:]
        return x_df, y_df
    
    def _splitTrainTest(self, x_df, y_df, train_ratio) :
        div_num = int(len(x_df.index)*train_ratio)
        x_train_df = x_df.iloc[:div_num,:]
        x_test_df = x_df.iloc[div_num:,:]
        y_train_df = y_df.iloc[:div_num,:]
        y_test_df = y_df.iloc[div_num:,:]
        return x_train_df, x_test_df, y_train_df, y_test_df

    def _minMaxNormalization(self, x_train_df, x_test_df) :
        mean = x_train_df.min()
        std = x_train_df.max() - x_train_df.min() + 0.00001
        x_train_normedDf = (x_train_df-mean)/std
        x_test_normedDf = (x_test_df-mean)/std
        return x_train_normedDf, x_test_normedDf

    def _standardzation(self, x_train_df, x_test_df) :
        mean = x_train_df.mean()
        std = x_train_df.std() + 0.00001
        x_train_normedDf = (x_train_df-mean)/std
        x_test_normedDf = (x_test_df-mean)/std
        return x_train_normedDf, x_test_normedDf

    def _toRNNList(self, x_train_df, y_train_df, x_test_df, y_test_df, numRecurrent) :

        x_train_list = x_train_df.values.tolist()
        y_train_list = y_train_df.values.tolist()
        x_test_list = x_test_df.values.tolist()
        y_test_list = y_test_df.values.tolist()
        testSetIndex = y_test_df.iloc[numRecurrent-1:,:].index

        rnn_train_x = []
        rnn_train_y = y_train_list[numRecurrent-1:]
        rnn_test_x = []
        rnn_test_y = y_test_list[numRecurrent-1:]

        for idx in range(numRecurrent, len(x_train_list)+1) :
            x_train_entry = x_train_list[idx-numRecurrent:idx]
            rnn_train_x.append(x_train_entry)

        for idx in range(numRecurrent, len(x_test_list)+1) :
            x_test_entry = x_test_list[idx-numRecurrent:idx]
            rnn_test_x.append(x_test_entry)

        return rnn_train_x, rnn_train_y, rnn_test_x, rnn_test_y, testSetIndex

    def _toDNNList(self, x_train_df, y_train_df, x_test_df, y_test_df) :

        testSetIndex = y_test_df.index
        x_train_list = x_train_df.values.tolist()
        y_train_list = y_train_df.values.tolist()
        x_test_list = x_test_df.values.tolist()
        y_test_list = y_test_df.values.tolist()
        return x_train_list, y_train_list, x_test_list, y_test_list, testSetIndex

    def _shuffleList(self, listX, listY) :
        tupleList = [(listX[i], listY[i]) for i in range(0, len(listY))]
        shuffle(tupleList)
        listX = [tupleList[i][0] for i in range(0, len(listY))]
        listY = [tupleList[i][1] for i in range(0, len(listY))]
        return listX, listY

    def getInputData(\
            self,\
            pathOfDNNinputData,\
            train_ratio,\
            num_input,\
            num_recurrence=0,\
            normalization_method="minmax",\
            data_randomization=True,\
            data_uniformalization=False,\
            RNN=False) :

        data_df = pd.read_csv(pathOfDNNinputData).set_index("datetime")
        data_df.index = pd.to_datetime(data_df.index)
        

        x_df, y_df = self._splitXY(data_df, num_input)

        if data_uniformalization == True :
            x_df = x_df.rank()

        x_train_df, x_test_df, y_train_df, y_test_df = self._splitTrainTest(x_df, y_df, train_ratio)

        if normalization_method == "minmax" :
            x_train_df, x_test_df = self._minMaxNormalization(x_train_df, x_test_df)
        elif normalization_method == "standarzation" :
            x_train_df, x_test_df = self._standardzation(x_train_df, x_test_df)

        if RNN == True :
            train_x, train_y, test_x, test_y, testSetIndex = self._toRNNList(x_train_df, y_train_df, x_test_df, y_test_df, num_recurrence)
        else :
            train_x, train_y, test_x, test_y, testSetIndex = self._toDNNList(x_train_df, y_train_df, x_test_df, y_test_df)

        if data_randomization == True :
            train_x, train_y= self._shuffleList(train_x, train_y)

        return train_x, train_y, test_x, test_y, testSetIndex
        

class NeuralNetwork :

    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.config = Configuration(LSTM=True)
            self.model = Model(self, LSTM=True)
            self.inputData = InputData(self.config.inputData, LSTM=True)

        if DNN == True :
            self.config = Configuration(DNN=True)
            self.model = Model(self, DNN=True)
            self.inputData = InputData(self.config.inputData, DNN=True)

        self.result = []
        if not os.path.exists(self.config.checkPoint.pathOfCheckpoint):
            os.makedirs(self.config.checkPoint.pathOfCheckpoint)

        if not os.path.exists(self.config.checkPoint.pathOfCheckpoint+"/"+sys.argv[0]):
            copy2(sys.argv[0], self.config.checkPoint.pathOfCheckpoint)
            
        

    def _makeMultipleIndependentDNNGraph(\
            self,\
            num_input=None,\
            num_label=None,\
            n_hidden_1=None,\
            n_hidden_2=None,\
            n_hidden_3=None,\
            learning_rate=None,\
            ) :

        if num_input == None :
            num_input=self.config.inputData.num_input
            num_label=self.config.inputData.num_label
            n_hidden_1=self.config.learning.n_hidden_1
            n_hidden_2=self.config.learning.n_hidden_2
            n_hidden_3=self.config.learning.n_hidden_3
            learning_rate=self.config.learning.learning_rate

        tf.reset_default_graph()
        g = tf.Graph()
        g.as_default()

        X = tf.placeholder(tf.float32, shape=(None, num_input))
        Y = tf.placeholder(tf.float32, shape=(None, num_label))
        keep_prob = tf.placeholder(tf.float32)

        initializer = tf.contrib.layers.xavier_initializer()

        weights = {
            'h1': [tf.Variable(initializer([num_input, n_hidden_1])) for i in range(0,num_label)],
            'h2': [tf.Variable(initializer([n_hidden_1, n_hidden_2])) for i in range(0,num_label)],
            'h3': [tf.Variable(initializer([n_hidden_2, n_hidden_3])) for i in range(0,num_label)],
            'out': [tf.Variable(initializer([n_hidden_3, 1])) for i in range(0,num_label)]
        }


        biases = {
            'b1': [tf.Variable(initializer([n_hidden_1,])) for i in range(0,num_label)],
            'b2': [tf.Variable(initializer([n_hidden_2,])) for i in range(0,num_label)],
            'b3': [tf.Variable(initializer([n_hidden_3,])) for i in range(0,num_label)],
            'out': [tf.Variable(initializer([1,])) for i in range(0, num_label)]
        }

        out_layer = []
        for i in range(0, num_label) :

            wx1 =tf.add(tf.matmul(X, weights['h1'][i]), biases['b1'][i])
            layer_1 = tf.nn.tanh(wx1)
            layer_1 = tf.nn.dropout(layer_1, keep_prob)

            wx2 = tf.add(tf.matmul(layer_1, weights['h2'][i]), biases['b2'][i])
            layer_2 = wx2
            layer_2 = tf.nn.tanh(layer_2)
           #layer_2 = tf.nn.leaky_relu(layer_2)
            layer_2 = tf.nn.dropout(layer_2, keep_prob)

            wx3 = tf.add(tf.matmul(layer_2, weights['h3'][i]), biases['b3'][i])
            layer_3 = wx3
            layer_3 = tf.nn.leaky_relu(wx3)
            layer_3 = tf.nn.dropout(layer_3, keep_prob)

            pred = tf.add(tf.matmul(layer_3, weights['out'][i]), biases['out'][i], name="pred")
           #pred = tf.add(tf.matmul(layer_2, weights['out'][i]), biases['out'][i], name="pred")
            out_layer.append(pred)

        Y_pred = tf.squeeze(tf.stack(out_layer, axis=1), 2)
        Y_pred = tf.nn.relu(Y_pred)
        train_op, loss_op, saver, optimizer = self._makeTrainOperator(Y_pred, Y)

        return train_op, loss_op, Y_pred, saver, X, Y, keep_prob, optimizer

    def _makeTrainOperator(self, Y_pred, Y) :
        learning_rate=self.config.learning.learning_rate
        loss_op = tf.reduce_mean(tf.pow(Y_pred-Y,2), 0)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, name="train_op")
        saver = tf.train.Saver()
        return train_op, loss_op, saver, optimizer

    def _makeSuperDeepMultipleIndependentDNNGraph(\
            self,\
            num_input=None,\
            num_label=None,\
            n_hidden_1=None,\
            n_hidden_2=None,\
            n_hidden_3=None,\
            learning_rate=None,\
            hiddenLayer=None
            ) :

        if num_input == None :
            num_input=self.config.inputData.num_input
            num_label=self.config.inputData.num_label
            n_hidden_1=self.config.learning.n_hidden_1
            n_hidden_2=self.config.learning.n_hidden_2
            n_hidden_3=self.config.learning.n_hidden_3
            learning_rate=self.config.learning.learning_rate
            hiddenLayer=self.config.learning.hiddenLayer
            n_hidden=self.config.learning.n_hidden

        tf.reset_default_graph()
        g = tf.Graph()
        g.as_default()

        X = tf.placeholder(tf.float32, shape=(None, num_input))
        Y = tf.placeholder(tf.float32, shape=(None, num_label))
        keep_prob = tf.placeholder(tf.float32)

        initializer = tf.contrib.layers.xavier_initializer()
        he_initializer = tf.contrib.layers.variance_scaling_initializer()

        weights = {
            'h1': [tf.Variable(initializer([num_input, n_hidden])) for i in range(0,num_label)],
            'out': [tf.Variable(initializer([n_hidden, 1])) for i in range(0,num_label)]
        }


        biases = {
            'b1': [tf.Variable(initializer([n_hidden,])) for i in range(0,num_label)],
            'out': [tf.Variable(initializer([1,])) for i in range(0, num_label)]
        }

        out_layer = []

        def _makeWX(inputLayer) :
            layer = object()
            for i in range(0, hiddenLayer) :
                if i != (hiddenLayer-1) :
                    weights = tf.Variable(he_initializer([n_hidden, n_hidden]))
                    biases = tf.Variable(he_initializer([n_hidden,]))
                    wx = tf.add(tf.matmul(inputLayer, weights), biases)
                    layer = tf.nn.leaky_relu(wx)
                else :
                    weights = tf.Variable(initializer([n_hidden, n_hidden]))
                    biases = tf.Variable(initializer([n_hidden,]))
                    wx = tf.add(tf.matmul(inputLayer, weights), biases)
                    layer = tf.nn.sigmoid(wx)
                layer = tf.nn.dropout(layer, keep_prob)

                inputLayer = layer
            return layer

        for i in range(0, num_label) :

            wx1 = tf.add(tf.matmul(X, weights['h1'][i]), biases['b1'][i])
            layer_1 = tf.nn.sigmoid(wx1)
            layer_1 = tf.nn.dropout(layer_1, keep_prob)

            last_layer = _makeWX(layer_1)

            pred = tf.add(tf.matmul(last_layer, weights['out'][i]), biases['out'][i], name="pred")
            #pred = tf.add(tf.matmul(layer_2, weights['out'][i]), biases['out'][i], name="pred")
            out_layer.append(pred)

        Y_pred = tf.squeeze(tf.stack(out_layer, axis=1), 2)
        train_op, loss_op, saver, optimizer = self._makeTrainOperator(Y_pred, Y)

        return train_op, loss_op, Y_pred, saver, X, Y, keep_prob, optimizer

    def _makeDeepLSTMGraph(\
            self,\
            seq_length=None,\
            input_dim=None,\
            output_dim=None,\
            hidden_dim=None,\
            learning_rate=None,\
            rnnMultiCellNum=None) :

        if seq_length == None :
            seq_length=self.config.inputData.numRecurrent
            input_dim=self.config.inputData.num_input
            output_dim=self.config.inputData.num_label
            hidden_dim=self.config.learning.rnnHiddenDim
            learning_rate=self.config.learning.learning_rate
            rnnMultiCellNum=self.config.learning.rnnMultiCellNum

        tf.reset_default_graph()
        g = tf.Graph()
        g.as_default()

        X = tf.placeholder(tf.float32, [None, seq_length, input_dim], name="X")
        Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        
        rnnLayerDimList = []
        if (rnnMultiCellNum == 2) :
            rnnLayerDimList = [output_dim]
        elif (rnnMultiCellNum == 3) :
            rnnLayerDimList = [output_dim**2, output_dim]
        elif (rnnMultiCellNum == 4) :
            rnnLayerDimList = [output_dim*4, output_dim*2, output_dim]
        elif (rnnMultiCellNum == 5) :
            rnnLayerDimList = [output_dim*8, output_dim*4, output_dim*2, output_dim]
        elif (rnnMultiCellNum == 6) :
            rnnLayerDimList = [output_dim*8, output_dim*4, output_dim*2, output_dim, output_dim]
        
        cell = []
        if (rnnMultiCellNum > 1) :
            print("this rnn model has %d stacked cells" % rnnMultiCellNum)
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=nn_hidden_dim, state_is_tuple=True, activation=tf.nn.sigmoid) for nn_hidden_dim in rnnLayerDimList[:2]]
            cells += [tf.contrib.rnn.BasicLSTMCell(num_units=nn_hidden_dim, state_is_tuple=True, activation=tf.nn.leaky_relu) for nn_hidden_dim in rnnLayerDimList[2:]]
            output_cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True, activation=tf.nn.leaky_relu)
            #output_cell = tf.nn.rnn_cell.DropoutWrapper(cell=output_cell, input_keep_prob=keep_prob)
            cells.append(output_cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else :
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True) 

        outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        
        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)
        train_op, loss_op, saver, optimizer = self._makeTrainOperator(Y_pred, Y)
        
        return train_op, loss_op, Y_pred, saver, X, Y, keep_prob, optimizer

    def _makeDeepLSTMGraph_workWellButIDontKnowWhy(\
            self,\
            seq_length=None,\
            input_dim=None,\
            output_dim=None,\
            hidden_dim=None,\
            learning_rate=None,\
            rnnMultiCellNum=None) :

        if seq_length == None :
            seq_length=self.config.inputData.numRecurrent
            input_dim=self.config.inputData.num_input
            output_dim=self.config.inputData.num_label
            hidden_dim=self.config.learning.rnnHiddenDim
            learning_rate=self.config.learning.learning_rate
            rnnMultiCellNum=self.config.learning.rnnMultiCellNum

        tf.reset_default_graph()
        g = tf.Graph()
        g.as_default()

        X = tf.placeholder(tf.float32, [None, seq_length, input_dim], name="X")
        Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        
        rnnLayerDimList = []
        if (rnnMultiCellNum == 2) :
            rnnLayerDimList = [output_dim]
        elif (rnnMultiCellNum == 3) :
            rnnLayerDimList = [output_dim**2, output_dim]
        elif (rnnMultiCellNum == 4) :
            rnnLayerDimList = [output_dim*4, output_dim*2, output_dim]
        elif (rnnMultiCellNum == 5) :
            rnnLayerDimList = [output_dim*8, output_dim*4, output_dim*2, output_dim]
        elif (rnnMultiCellNum == 6) :
            rnnLayerDimList = [output_dim*8, output_dim*4, output_dim*2, output_dim, output_dim]
        
        cell = []
        if (rnnMultiCellNum > 1) :
            print("this rnn model has %d stacked cells" % rnnMultiCellNum)
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=nn_hidden_dim, state_is_tuple=True, activation=tf.nn.sigmoid) for nn_hidden_dim in rnnLayerDimList[:2]]
            cells += [tf.contrib.rnn.BasicLSTMCell(num_units=nn_hidden_dim, state_is_tuple=True, activation=tf.nn.leaky_relu) for nn_hidden_dim in rnnLayerDimList[2:]]
            output_cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True, activation=tf.nn.leaky_relu)
            #output_cell = tf.nn.rnn_cell.DropoutWrapper(cell=output_cell, input_keep_prob=keep_prob)
            cells.append(output_cell)
            _dropoout_cells_unused_but_workWell = [tf.nn.rnn_cell.DropoutWrapper(cell=rnn_cell, output_keep_prob=keep_prob) for rnn_cell in cells]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else :
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True) 

        outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        
        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)
        train_op, loss_op, saver, optimizer = self._makeTrainOperator(Y_pred, Y)
        
        return train_op, loss_op, Y_pred, saver, X, Y, keep_prob, optimizer

    def _makeSimpleLSTMGraph(\
            self,\
            seq_length=None,\
            input_dim=None,\
            output_dim=None,\
            hidden_dim=None,\
            learning_rate=None,\
            rnnMultiCellNum=None) :

        if seq_length == None :
            seq_length=self.config.inputData.numRecurrent
            input_dim=self.config.inputData.num_input
            output_dim=self.config.inputData.num_label
            hidden_dim=self.config.learning.rnnHiddenDim
            learning_rate=self.config.learning.learning_rate
            rnnMultiCellNum=self.config.learning.rnnMultiCellNum

        tf.reset_default_graph()
        g = tf.Graph()
        g.as_default()

        X = tf.placeholder(tf.float32, [None, seq_length, input_dim], name="X")
        Y = tf.placeholder(tf.float32, [None, output_dim], name="Y")
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        
        cell = [] #empty object
        if (rnnMultiCellNum > 1) :
            print("this rnn model has %d stacked cells" % rnnMultiCellNum)
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True, activation=tf.nn.leaky_relu) for i in range(rnnMultiCellNum)]
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else :
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True) 

        outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        
        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)
        train_op, loss_op, saver, optimizer = self._makeTrainOperator(Y_pred, Y)
        
        return train_op, loss_op, Y_pred, saver, X, Y, keep_prob, optimizer

    def doTraining(\
            self,\
            trainX=None,\
            trainY=None,\
            testX=None,\
            testY=None,\
            x_placeholder=None,\
            y_placeholder=None,\
            keep_prob=None,\
            train_op=None,\
            loss_op=None,\
            Y_pred_op=None,\
            saver=None,\
            howManyEpoch=None,\
            display_step=None,\
            output_keep_prob=None,\
            input_keep_prob=None,\
            save_step=None,\
            pathOfCheckpoint=None,\
            batchDivider=None,\
            filenameOfCheckpoint=None) :

        if trainX == None :
            trainX=self.inputData.train_x
            trainY=self.inputData.train_y
            testX=self.inputData.test_x
            testY=self.inputData.test_y
            x_placeholder=self.model.X
            y_placeholder=self.model.Y
            keep_prob=self.model.keep_prob
            train_op=self.model.train_op
            loss_op=self.model.loss_op
            Y_pred_op=self.model.Y_pred_op
            saver=self.model.saver
            howManyEpoch=self.config.learning.numLearningEpoch
            display_step=self.config.learning.display_step
            input_keep_prob=self.config.learning.input_keep_prob
            save_step=self.config.checkPoint.save_step
            pathOfCheckpoint=self.config.checkPoint.pathOfCheckpoint
            batchDivider=self.config.learning.batchDivider
            filenameOfCheckpoint=self.config.checkPoint.filenameOfCheckpoint

        init_step = 0

        with tf.Session() as sess :
            
            sess.run(tf.global_variables_initializer())

            #resotre check point
            ckpt_path = tf.train.latest_checkpoint(pathOfCheckpoint)
            if ckpt_path :
                saver.restore(sess, ckpt_path)
                init_step = int(ckpt_path.rsplit("-")[1])
            
            learningRateModificationCount = 0
            for step in range(init_step, howManyEpoch) :

                if ((step % display_step) == 0) :
                    loss, cur_lr = sess.run([loss_op, self.model.optimizer._lr_t], feed_dict={x_placeholder: trainX, y_placeholder: trainY, keep_prob: input_keep_prob})
                    testPredict = sess.run(Y_pred_op, feed_dict={x_placeholder: testX, y_placeholder: testY, keep_prob: 1.0})
                    print("Epoch "+str(step)+", cost = ", loss, ("learningRate : %.6f" % (cur_lr)))
                    self._modelEvaluation(predList=testPredict, labelList=testY)
                    if step == (howManyEpoch-1) :
                        break

                if ((step % save_step) == 0) :
                    print("save current state")
                    saver.save(sess, pathOfCheckpoint+filenameOfCheckpoint, global_step=step)

                self._batchTrainer(sess=sess,\
                        train_op=train_op,\
                        batch_divider=batchDivider,\
                        trainX=trainX,\
                        trainY=trainY,\
                        x_placeholder=x_placeholder,\
                        y_placeholder=y_placeholder,\
                        keep_prob=keep_prob,\
                        output_keep_prob=output_keep_prob,\
                        input_keep_prob=input_keep_prob)

                
            self.result = sess.run(Y_pred_op, feed_dict={x_placeholder: testX, keep_prob: 1.0})
            return self.result

    def getResult(self) :
        testPredict = []
        with tf.Session() as sess :
            testPredict = sess.run(self.model.Y_pred_op, feed_dict={self.model.X: self.inputData.test_x, self.model.keep_prob: 1.0})
        return testPredict

    def getResult(self, inputList) :
        testPredict = []
        with tf.Session() as sess :
            testPredict = sess.run(self.model.Y_pred_op, feed_dict={self.model.X: inputList, self.model.keep_prob: 1.0})
        return testPredict

    def _batchTrainer(self, sess, train_op, batch_divider, trainX, trainY, x_placeholder, y_placeholder, keep_prob, output_keep_prob, input_keep_prob) :
        batch_size = len(trainX)//batch_divider+1
        x_batch = []
        y_batch = []
        i = 0

        while (i < len(trainX)) :
            x_batch.append(trainX[i])
            y_batch.append(trainY[i])
            if ((i+1) % batch_size == 0 or i == len(trainX) - 1) :
                _ = sess.run(train_op, feed_dict={x_placeholder: x_batch, y_placeholder: y_batch, keep_prob: input_keep_prob})
                x_batch = []
                y_batch = []
            i += 1

    def saveResultAsCSV(self,\
            result=None,\
            testY=None) :

        if result == None :
            result=self.result
            testY=self.inputData.test_y
            testSetIndex=self.inputData.testSetIndex

        df = pd.DataFrame(index=testSetIndex)
        for i in range(0, len(result[0])) :
            df["pred_"+str(i)] = [entry[i] for entry in result]
        for i in range(0, len(testY[0])) :
            df["label_"+str(i)] = [entry[i] for entry in testY]
        
        df = df.join(self.inputData.originDf["D.DeRatedPower"])
        df.to_csv(self.config.learning.resultPath)

    def _modelEvaluation(self, predList, labelList) :
        predDfList = []
        labelDfList = []
        for column_idx in range(len(predList[0])) :
            predDf = pd.DataFrame()
            labelDf = pd.DataFrame()
            predDf["value"] = [row[column_idx] for row in predList]
            labelDf["value"] = [row[column_idx] for row in labelList]
            predDfList.append(predDf)
            labelDfList.append(labelDf)
        toPrint = ""
        labelList = self.config.inputData.labelList
        for idx in range(len(predDfList)) :
            toPrint += ("-"*43 + "  %s  " + "-"*43+"\n") % labelList[idx]
            toPrint += "%20s | %20s | %9s | %9s | %9s | %24s\n" % ("base percentage", "underbase value", "PE", "RMSE", "MAE", "10% inner count ratio")
            toPrint += self._reportAccuracy(predDfList[idx], labelDfList[idx])
        print(toPrint)

    def _reportAccuracy(self, predDf, labelDf) :
        toPrint = ""
        accuracyTupleList = self._getAccuracyConsideringPercentile(predDf, labelDf)
        for tupleEntry in accuracyTupleList :
            if(tupleEntry[0] % 10 == 0) :
                toPrint += ("%19d%% | %20.4f | %8.4f%% | %8.4f%% | %8.4f%% | %23.4f%%\n" % tupleEntry)
        return toPrint
        
        
    def _getAccuracyConsideringPercentile(self, predDf, labelDf) :
        accuracyList = []
        percentileList = [0.01*idx for idx in range(0, 100)]
        for percent in percentileList :
            percentileValue = labelDf.quantile(percent)
            srcDf = labelDf[labelDf > percentileValue]
            dstDf = predDf[labelDf > percentileValue]
            PE = ((srcDf-dstDf)/srcDf).abs().mean()*100
            RMSE = sqrt(((dstDf-srcDf)**2).mean())/srcDf.mean()*100
            MAE = ((dstDf-srcDf).abs().mean())/srcDf.mean()*100
            underBoundary = srcDf*0.9
            upperBoundary = srcDf*1.1
            countRatio = dstDf[(underBoundary < dstDf) & (dstDf < upperBoundary)].count()/dstDf.count()*100
            accuracyTuple = (int(percent*100), percentileValue, PE, RMSE, MAE, countRatio)
            accuracyList.append(accuracyTuple)
        return accuracyList

def main() :
    RNN = NeuralNetwork(DNN=True)
    RNN.doTraining()
    RNN.saveResultAsCSV()

main()
