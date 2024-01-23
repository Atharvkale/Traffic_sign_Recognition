import numpy as np
import pandas as pd
from PIL import Image
import os
import seaborn as sns
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


train_data = []
train_labels = []
classes = 43

#Retrieving the images and their labels 
for i in range(classes):
    path = "D:\\AI_CP\\Dataset\\Train\\" + str(i)
    images = os.listdir(path)
    for a in images:
        try:
            image = Image.open(path + '\\'+ a)
            image = image.resize((30,30))
            image = np.array(image)
            train_data.append(image)
            train_labels.append(i)
        except:
            print("Error loading image")
#Converting lists into numpy arrays
train_data = np.array(train_data)
train_labels = np.array(train_labels)

train_data_flattened = []

for i in train_data:
    train_data_flattened.append(i.flatten())
    
#np.save("F:\\VIT\\SY\\2nd sem\\EDI\\Training_data_array",train_data)
#np.save("F:\\VIT\\SY\\2nd sem\\EDI\\Training_data_array_flattened",train_data_flattened)
#np.save("F:\\VIT\\SY\\2nd sem\\EDI\\Training_labels_array",train_labels)


test_data = []
path = "D:\\AI_CP\\Dataset\\Test"
images = os.listdir(path)
for i in images[:-1]:
        try:
            image = Image.open(path + '\\'+ i)
            image = image.resize((30,30))
            image = np.array(image)
            test_data.append(image)
        except:
            print("Error loading image")

test_data_flattened = []

for i in test_data:
    test_data_flattened.append(i.flatten())

#np.save("F:\\VIT\\SY\\2nd sem\\EDI\\Testing_data_array",test_data)
#np.save("F:\\VIT\\SY\\2nd sem\\EDI\\Testing_data_array_flattened",test_data_flattened)




#train_data_flattened = np.load("F:\\VIT\\SY\\2nd sem\\EDI\\Training_data_array_flattened.npy")
#train_labels = np.load("F:\\VIT\\SY\\2nd sem\\EDI\\Training_labels_array.npy")
#test_data_flattened = np.load("F:\\VIT\\SY\\2nd sem\\EDI\\Testing_data_array_flattened.npy")



classifier = KNeighborsClassifier(n_neighbors=198)
classifier.fit(train_data_flattened, train_labels)

predicted_values = classifier.predict(test_data_flattened)
#predicted_values = np.load("F:\\VIT\\SY\\2nd sem\\EDI\\Predicted_values_with_knn.npy")


df=pd.read_csv("D:\\AI_CP\\Dataset\\Test.csv")
x=[]
x.append(df["ClassId"])

report=classification_report(x[0],predicted_values)
print("The report of KNN",report)

acc=metrics.accuracy_score(x[0],predicted_values)
print("The accuracy with KNN:",acc*100,"%")




classifier1 = LogisticRegression()
classifier1.fit(train_data_flattened, train_labels)

#pickle.dump(classifier1, open('classifier.pkl', 'wb'))
#pickled_model1 = pickle.load(open('F:\\VIT\\SY\\2nd sem\\EDI\\classifier.pkl', 'rb'))


predicted_values1 = classifier1.predict(test_data_flattened)
#predicted_values1 = np.load("F:\\VIT\\SY\\2nd sem\\EDI\\Predicted_values_with_logistic_regression.npy")

report1=classification_report(x[0],predicted_values1)
print("The report of LR",report1)

confusion1 = confusion_matrix(x[0],predicted_values1)

s1=sns.heatmap(confusion1,annot=True,cmap="nipy_spectral_r")
print(s1.set_title("CONFUSION MATRIX LR"))


acc1=metrics.accuracy_score(x[0],predicted_values1)
print("The accuracy with LR:",acc1*100,"%")

#Random Forest
classifier2=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=0)
classifier2.fit(train_data_flattened, train_labels)

predicted_values2 = classifier2.predict(test_data_flattened)
#predicted_values1 = np.load("F:\\VIT\\SY\\2nd sem\\EDI\\Predicted_values_with_logistic_regression.npy")

report2=classification_report(x[0],predicted_values2)
print("The report of RF",report2)

acc2=metrics.accuracy_score(x[0],predicted_values2)
print("The accuracy with RF:",acc2*100,"%")


#dictionary to label all traffic signs class.
classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }



try:
    image = Image.open(input("Enter image address: "))
    image = image.resize((30,30))
    image = np.array(image)
    image= image.flatten()
    
    #classifier = pickle.load(open('F:\\VIT\\SY\\2nd sem\\EDI\\classifier.pkl', 'rb'))
    
    class_of_image = classifier.predict([image])
    
    print("\nTraffic sign : ",classes[class_of_image[0]], "\n\n")
    
except:
    print("Error in loading image...")