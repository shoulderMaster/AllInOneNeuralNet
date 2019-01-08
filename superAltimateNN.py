import tensorflow as tf
import pandas as pd
import os
from random import shuffle

'''
SupurPowerElegantlyAutomaticIndepententIndividualMultipleRNN
    config
        learning
            resultPath
            batchDivider
            learning_rate
            dropoutRate
            output_keep_prob
            rnnHiddenDim
            rnnMultiCellNum
            numLearningEpoch
            display_step

        checkPoint
            pathOfCheckpoint
            filenameOfCheckpoint
            save_step

        inputData
            pathOfRNNinputData
            num_input
            num_label
            train_ratio
            numRecurrent
    model
        train_op
        loss_op
        Y_pred_op
        saver
        X
        Y
        keep_prob

    inputData
        train_x
        train_y
        test_x
        test_y

    result
    doTraining()
    getResult()
    saveResultAsCSV()
'''

class CheckPointConfiguration :
    def __init__(self) :
        #self.pathOfCheckpoint = "./model_export/simulater_190108_dropout30_3hiddenLayer_512"
        self.pathOfCheckpoint = "./model_export/simulater_190108_dropout10_4stacked_6recurrent"
        self.filenameOfCheckpoint = "/model_data"
        self.save_step= 200


class InputDataConfiguration :
    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.pathOfinputData = "./RNN_input_data.csv"
            self.num_input = 7
            self.num_label = 3
            self.train_ratio = 0.7
            self.numRecurrent = 8

        if DNN == True :
            self.pathOfinputData = "./DNN_input_data.csv"
            self.num_input = 7
            self.num_label = 3
            self.train_ratio = 0.7

class LearningConfiguration :
    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.resultPath = "result_190108_dropout10_4stacked_6recurrent.csv"
            self.batchDivider = 8
            self.learning_rate = 0.05
            self.dropoutRate = 0.1
            self.output_keep_prob = 1 - self.dropoutRate
            self.input_keep_prob = 1 - self.dropoutRate
            self.rnnHiddenDim = 64
            self.rnnMultiCellNum = 4
            self.numLearningEpoch = 2000
            self.display_step = 30

        if DNN == True :
            self.resultPath = "result_190108_dropout30_3hiddenLayer_512.csv"
            self.batchDivider = 8
            self.learning_rate = 0.05
            self.dropoutRate = 0.1
            self.input_keep_prob = 1 - self.dropoutRate
            self.numLearningEpoch = 2000
            self.display_step = 30
            self.n_hidden_1 = 512
            self.n_hidden_2 = 512
            self.n_hidden_3 = 512

class Configuration :
    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.learning = LearningConfiguration(LSTM=True)
            self.inputData = InputDataConfiguration(LSTM=True)
            self.checkPoint = CheckPointConfiguration()

        if DNN == True :
            self.learning = LearningConfiguration(DNN=True)
            self.inputData = InputDataConfiguration(DNN=True)
            self.checkPoint = CheckPointConfiguration()


class Model :
    def __init__(self,NN, LSTM=None, DNN=None) :

        if LSTM == True :
            self.train_op, self.loss_op, self.Y_pred_op, self.saver, self.X, self.Y, self.keep_prob\
            = NN._makeMultipleIndependentLSTMGraph()

        if DNN == True :
            self.train_op, self.loss_op, self.Y_pred_op, self.saver, self.X, self.Y, self.keep_prob\
            = NN._makeMultipleIndependentDNNGraph()

class InputData : 
    
    def __init__(self, inputDataConfiguation, LSTM=None, DNN=None) :

        if LSTM == True :
            self.train_x, self.train_y, self.test_x, self.test_y\
            = self.getRNNInputData(\
                    pathOfRNNinputData=inputDataConfiguation.pathOfinputData,\
                    train_ratio=inputDataConfiguation.train_ratio,\
                    numRecurrent=inputDataConfiguation.numRecurrent,\
                    num_input=inputDataConfiguation.num_input)

        if DNN == True :
            self.train_x, self.train_y, self.test_x, self.test_y\
            = self.getDNNInputData(\
                    pathOfDNNinputData=inputDataConfiguation.pathOfinputData,\
                    train_ratio=inputDataConfiguation.train_ratio,\
                    num_input=inputDataConfiguation.num_input)
        

    def getRNNInputData(\
            self,\
            pathOfRNNinputData,\
            train_ratio,\
            numRecurrent,\
            num_input) :

        data_df = pd.read_csv(pathOfRNNinputData).set_index("datetime")
        data_df.index = pd.to_datetime(data_df.index)

        div_num = int(len(data_df.index)*train_ratio)

        train_df = data_df.iloc[:div_num,:]
        test_df = data_df.iloc[div_num:,:]

        x_train_df = train_df.iloc[:,:num_input]
        x_test_df = test_df.iloc[:,:num_input]
        y_train_df = train_df.iloc[:,num_input:]
        y_test_df = test_df.iloc[:,num_input:]

        mean = x_train_df.mean()
        std = x_train_df.std() + 0.00001

        x_train_list = ((x_train_df-mean)/std).values.tolist()
        y_train_list = y_train_df.values.tolist()
        x_test_list = ((x_test_df-mean)/std).values.tolist()
        y_test_list = y_test_df.values.tolist()

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

        if(len(rnn_train_x) != len(rnn_train_y)) :
            print("shape of input data is not incompatible")
        rnn_train_x, rnn_train_y = self._shuffleList(rnn_train_x, rnn_train_y)

        return rnn_train_x, rnn_train_y, rnn_test_x, rnn_test_y
    
    def getDNNInputData(\
            self,\
            pathOfDNNinputData,\
            train_ratio,\
            num_input) :

        data_df = pd.read_csv(pathOfDNNinputData).set_index("datetime")
        data_df.index = pd.to_datetime(data_df.index)

        div_num = int(len(data_df.index)*train_ratio)

        train_df = data_df.iloc[:div_num,:]
        test_df = data_df.iloc[div_num:,:]

        x_train_df = train_df.iloc[:,:num_input]
        x_test_df = test_df.iloc[:,:num_input]
        y_train_df = train_df.iloc[:,num_input:]
        y_test_df = test_df.iloc[:,num_input:]

        mean = x_train_df.mean()
        std = x_train_df.std() + 0.00001

        x_train_list = ((x_train_df-mean)/std).values.tolist()
        y_train_list = y_train_df.values.tolist()
        x_test_list = ((x_test_df-mean)/std).values.tolist()
        y_test_list = y_test_df.values.tolist()

        x_train_list, y_train_list= self._shuffleList(x_train_list, y_train_list)

        return x_train_list, y_train_list, x_test_list, y_test_list

    def _shuffleList(self, listX, listY) :
        tupleList = [(listX[i], listY[i]) for i in range(0, len(listY))]
        shuffle(tupleList)
        listX = [tupleList[i][0] for i in range(0, len(listY))]
        listY = [tupleList[i][1] for i in range(0, len(listY))]
        return listX, listY


class SupurPowerElegantAutomaticNeuralNetwork :

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

            # Hidden fully connected layer with 512 neurons
           #wx1 = tf.matmul(x, weights['h1'][i])
            wx1 =tf.add(tf.matmul(X, weights['h1'][i]), biases['b1'][i])
           #bn1 = tf.layers.batch_normalization(
           #    inputs=wx1,
           #    momentum=0.9,
           #    epsilon=0.001,
           #    center=True,
           #    scale=True,
           #    training = is_training,
           #)

            layer_1 = tf.nn.sigmoid(wx1)
            layer_1 = tf.nn.dropout(layer_1, keep_prob)

            # Hidden fully connected layer with 512 neurons
           #wx2 = tf.matmul(layer_1, weights['h2'][i])
            wx2 = tf.add(tf.matmul(layer_1, weights['h2'][i]), biases['b2'][i])
           #bn2 = tf.layers.batch_normalization(
           #    inputs=wx2,
           #    momentum=0.9,
           #    epsilon=0.001,
           #    center=True,
           #    scale=True,
           #    training = is_training,
           #)
            layer_2 = tf.nn.sigmoid(wx2)
           #layer_2 = tf.add(tf.matmul(layer_1, weights['h2'][i]), biases['b2'][i])
            layer_2 = tf.nn.dropout(layer_2, keep_prob)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h3'][i]), biases['b3'][i])
           #layer_3 = tf.nn.sigmoid(layer_3)
            layer_3 = tf.nn.dropout(layer_3, keep_prob)

            out_layer += [tf.add(tf.matmul(layer_3, weights['out'][i]), biases['out'][i], name="pred")]
        #make prediction tensor with shape (none, 46)
        Y_pred = tf.squeeze(tf.stack(out_layer, axis=1), 2)

        loss_op = tf.reduce_mean(tf.pow(Y_pred-Y,2), 0)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op, name="train_op")
        saver = tf.train.Saver()

        return train_op, loss_op, Y_pred, saver, X, Y, keep_prob

    def _makeMultipleIndependentLSTMGraph(\
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
        
        out_layer = []
        for i in range(output_dim) :
            cell = [] #empty object
            if (rnnMultiCellNum > 1) :
                cells = [tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True, activation=tf.nn.leaky_relu) for i in range(rnnMultiCellNum-1)]
                output_cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True, activation=tf.nn.leaky_relu)
                output_cell = tf.nn.rnn_cell.DropoutWrapper(cell=output_cell, input_keep_prob=keep_prob)
                cells.append(output_cell)
                cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
            else :
                cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True) 

            outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
            Y_pred_cell = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)
            out_layer.append(Y_pred_cell)

        Y_pred = tf.squeeze(tf.stack(out_layer, axis=1), 2)
        loss = tf.reduce_mean(tf.square(Y_pred-Y))
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        saver = tf.train.Saver()
        
        return train, loss, Y_pred, saver, X, Y, keep_prob

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
            cells = [tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True, activation=tf.nn.leaky_relu) for i in range(rnnMultiCellNum-1)]
            output_cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True, activation=tf.nn.leaky_relu)
            output_cell = tf.nn.rnn_cell.DropoutWrapper(cell=output_cell, input_keep_prob=keep_prob)
            cells.append(output_cell)
            cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else :
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=output_dim, state_is_tuple=True) 

        outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
        
        Y_pred = tf.contrib.layers.fully_connected(outputs[:,-1], output_dim, activation_fn=None)
        loss = tf.reduce_mean(tf.square(Y_pred-Y))
        
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train = optimizer.minimize(loss)

        saver = tf.train.Saver()
        
        return train, loss, Y_pred, saver, X, Y, keep_prob

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
            
            for step in range(init_step, howManyEpoch+1) :

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

                if ((step % display_step) == 0 ) :
                    loss = sess.run(loss_op, feed_dict={x_placeholder: trainX, y_placeholder: trainY, keep_prob: input_keep_prob})
                    testPredict = sess.run(Y_pred_op, feed_dict={x_placeholder: testX, keep_prob: 1.0})
                    print("Step "+str(step)+", cost = ", loss)
                    self._modelEvaluation(predList=testPredict, labelList=testY)

                if ((step % save_step) == 0) :
                    print("save current state")
                    saver.save(sess, pathOfCheckpoint+filenameOfCheckpoint, global_step=step)
                
            self.result = sess.run(Y_pred_op, feed_dict={x_placeholder: testX, keep_prob: 1.0})
            return self.result

    def getResult(self) :
        testPredict = []
        with tf.Session() as sess :
            testPredict = sess.run(self.model.Y_pred_op, feed_dict={self.model.X: self.inputData.test_x, self.model.keep_prob: 1.0})
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

        df = pd.DataFrame()
        for i in range(0, len(result[0])) :
            df["pred_"+str(i)] = [entry[i] for entry in result]
        for i in range(0, len(testY[0])) :
            df["label_"+str(i)] = [entry[i] for entry in testY]
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
        labelList = ["Rotor Speed (RPM)", "Active Power (W)", "Generator Speed (RPM)"]
        for idx in range(len(predDfList)) :
            toPrint += ("-"*32 + "  %s  " + "-"*32+"\n") % labelList[idx]
            toPrint += "%20s | %20s | %12s | %24s\n" % ("base percentage", "underbase value", "deviation", "10% inner count ratio")
            toPrint += self._reportAccuracy(predDfList[idx], labelDfList[idx])
        print(toPrint)

    def _reportAccuracy(self, predDf, labelDf) :
        toPrint = ""
        accuracyTupleList = self._getAccuracyConsideringPercentile(predDf, labelDf)
        for tupleEntry in accuracyTupleList :
            if(tupleEntry[0] % 10 == 0) :
                toPrint += ("%19d%% | %20.4f | %11.4f%% | %23.4f%%\n" % tupleEntry)
        return toPrint
        
        
    def _getAccuracyConsideringPercentile(self, predDf, labelDf) :
        accuracyList = []
        percentileList = [0.01*idx for idx in range(0, 100)]
        for percent in percentileList :
            percentileValue = labelDf.quantile(percent)
            srcDf = labelDf[labelDf > percentileValue]
            dstDf = predDf[labelDf > percentileValue]
            deviation = ((dstDf-srcDf)/srcDf).abs().mean()*100
            underBoundary = srcDf*0.9
            upperBoundary = srcDf*1.1
            countRatio = dstDf[(underBoundary < dstDf) & (dstDf < upperBoundary)].count()/dstDf.count()*100
            accuracyTuple = (int(percent*100), percentileValue, deviation, countRatio)
            accuracyList.append(accuracyTuple)
        return accuracyList

def main() :
    RNN = SupurPowerElegantAutomaticNeuralNetwork(DNN=True)
    RNN.doTraining()
    RNN.saveResultAsCSV()

main()
