

class CheckPointConfiguration :
    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.pathOfCheckpoint = "./model_export/removeActivePower_recurrence4"
            self.filenameOfCheckpoint = "/model_data"
            self.save_step= 200

        if DNN == True :
            self.pathOfCheckpoint = "./model_export/removeActivePower_dnn_noPred"
            self.filenameOfCheckpoint = "/model_data"
            self.save_step= 30


class InputDataConfiguration :
    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.pathOfinputData = "./RNN_input_data_withoutActivePower.csv"
            self.num_input = 5
            self.num_label = 2
            self.train_ratio = 0.7
            self.numRecurrent = 3
            self.labelList = ["Active Power (W)", "Generator Speed (RPM)"]

        if DNN == True :
            self.pathOfinputData = "./RNN_input_data.csv"
            self.num_input = 5
            self.num_label = 1
            self.train_ratio = 0.7
            #self.labelList = ["Active Power (W)", "Derated Power (W)", "Generator Speed (RPM)"]
            self.labelList = ["Active Power (W)", "Generator Speed (RPM)"]

class LearningConfiguration :
    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.resultPath = "result_LSTM.csv"
            self.batchDivider = 8
            self.learning_rate = 0.05
            self.dropoutRate = 0.0
            self.output_keep_prob = 1 - self.dropoutRate
            self.input_keep_prob = 1 - self.dropoutRate
            self.rnnHiddenDim = 64
            self.rnnMultiCellNum = 4
            self.numLearningEpoch = 1020
            self.display_step = 30

        if DNN == True :
            self.resultPath = "result_DNN.csv"
            self.batchDivider = 8
            self.learning_rate = 0.001
            self.dropoutRate = 0.2
            self.input_keep_prob = 1 - self.dropoutRate
            self.numLearningEpoch = 1000+1
            self.display_step = 20
            self.n_hidden_1 = self.n_hidden_2 = self.n_hidden_3 = 16
            self.hiddenLayer = 10
            self.n_hidden = 512


class Configuration :
    def __init__(self, LSTM=None, DNN=None) :

        if LSTM == True :
            self.learning = LearningConfiguration(LSTM=True)
            self.inputData = InputDataConfiguration(LSTM=True)
            self.checkPoint = CheckPointConfiguration(LSTM=True)

        if DNN == True :
            self.learning = LearningConfiguration(DNN=True)
            self.inputData = InputDataConfiguration(DNN=True)
            self.checkPoint = CheckPointConfiguration(DNN=True)
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
