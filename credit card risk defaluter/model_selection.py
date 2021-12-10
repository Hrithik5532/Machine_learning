from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

class Model_selection():
    def best_model(X,y):
        X_train, X_val, y_train, y_val= train_test_split(X,y,test_size = 0.3,random_state=100)
        
        def logisticRegression():
            logmodel = LogisticRegression()
            logmodel.fit(X_train, y_train)
            prediction = logmodel.predict(X_val)
            global logisticRegression_score
            logisticRegression_score = (100*f1_score(y_val,prediction,average="macro"))

        def decisionTree():
            decisiontree_model = DecisionTreeClassifier(criterion='entropy',max_depth=15)
            decisiontree_model.fit(X_train,y_train)
            prediction = decisiontree_model.predict(X_val)
            global decisionTree_score
            decisionTree_score = (100*f1_score(y_val, prediction,average='macro'))
        def ann():
            callback = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
            ann_model= Sequential()
            ann_model.add(Dense(16,activation="relu"))
            ann_model.add(Dropout(0.3))
            ann_model(Dense(8,activation="relu"))
            ann_model.add(Dropout(0.2))
            ann_model.add(Dense(1,activation="sigmoid"))
            ann_model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
            ann_model.fit(X_train,y_train,batch_size=63,epochs=100,verbose=1,validation_data=[X_val,y_val],callbacks=callback)
            prediction = ann_model.predict(X_val)
            global Ann_score
            Ann_score = (100*f1_score(y_val, prediction,average='macro'))
            
        logisticRegression()
        decisionTree()

        best_score = max(logisticRegression_score,decisionTree_score,)
        if best_score == logisticRegression_score:
            return best_score , "LogisticRegression is Best for this dataset"
        else:
            return best_score , "DecisionTree is Best for this dataset"
