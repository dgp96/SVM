from libsvm.svmutil import *
import scipy.io
train_data = scipy.io.loadmat("trainData.mat")
test_data = scipy.io.loadmat("testData.mat")

x1_train = train_data['X1']
x2_train = train_data['X2']
x3_train = train_data['X3']
y_train_flatten = train_data['Y'].flatten()

x1_test = test_data['X1']
x2_test = test_data['X2']
x3_test = test_data['X3']
y_test_flatten = test_data['Y'].flatten()

param = svm_parameter('-c 10 -t 0')
#Step 0 part1, training model_step0_que1_case1 1 indicates training with X1 only and so on
model_step0_que1_case1 = svm_train(y_train_flatten,x1_train, '-c 10 -t 0')
model_step0_que1_case2 = svm_train(y_train_flatten,x2_train, '-c 10 -t 0')
model_step0_que1_case3 = svm_train(y_train_flatten,x3_train, '-c 10 -t 0')

#Step 0 part1 predict
p_label_step0_que1_case1, p_acc_step0_que1_case1, p_val_step0_que1_case1 = svm_predict(y_test_flatten,x1_test, model_step0_que1_case1)
p_label_step0_que1_case2, p_acc_step0_que1_case2, p_val_step0_que1_case2 = svm_predict(y_test_flatten,x2_test, model_step0_que1_case2)
p_label_step0_que1_case3, p_acc_step0_que1_case3, p_val_step0_que1_case3 = svm_predict(y_test_flatten,x3_test, model_step0_que1_case3)

#Accuracies
print("Accuracy for Step 0 part 1 and feature vector X1:" + str(p_acc_step0_que1_case1))
print("Accuracy for Step 0 part 1 and feature vector X2:" + str(p_acc_step0_que1_case2))
print("Accuracy for Step 0 part 1 and feature vector X3:" + str(p_acc_step0_que1_case3))

#Step 0 part2, training model_step0_que2_case1 indicates training with X1 only and '-b 1' indicates generating probabilities for each class and so on
model_step0_que2_case1 = svm_train(y_train_flatten,x1_train, '-c 10 -t 0 -b 1')
model_step0_que2_case2 = svm_train(y_train_flatten,x2_train, '-c 10 -t 0 -b 1')
model_step0_que2_case3 = svm_train(y_train_flatten,x3_train, '-c 10 -t 0 -b 1')

#Step 0 part 2 predict
p_label_step0_que2_case1, p_acc_step0_que2_case1, p_val_step0_que2_case1 = svm_predict(y_test_flatten,x1_test, model_step0_que2_case1,'-b 1')
p_label_step0_que2_case2, p_acc_step0_que2_case2, p_val_step0_que2_case2 = svm_predict(y_test_flatten,x2_test, model_step0_que2_case2,'-b 1')
p_label_step0_que2_case3, p_acc_step0_que2_case3, p_val_step0_que2_case3 = svm_predict(y_test_flatten,x3_test, model_step0_que2_case3,'-b 1')

print("Accuracy for Step 0 part 2 and feature vector X1:" + str(p_acc_step0_que2_case1))
print("Accuracy for Step 0 part 2 and feature vector X2:" + str(p_acc_step0_que2_case2))
print("Accuracy for Step 0 part 2 and feature vector X3:" + str(p_acc_step0_que2_case3))

#Step1 : Predicting the classes with maximum probability
import numpy as np
length1 = len(p_val_step0_que2_case1) #no of samples
length2 = len(p_val_step0_que2_case1[0]) #no of classes
#print(length1)
#print(length2)
predicted_class = []
for i in range(length1):
    avg=[]
    for j in range(length2):
        avg.append((p_val_step0_que2_case1[i][j]+p_val_step0_que2_case2[i][j]+p_val_step0_que2_case3[i][j])/3)
    max_class = np.argmax(avg)
    predicted_class.append(max_class+1)
print(predicted_class[:10])

#Calculating the accuracy
correct = 0
for i in range(len(predicted_class)):
    if predicted_class[i] == y_test_flatten[i]:
        correct+=1
print("Accuracy for Step1: "+str((correct/len(predicted_class))*100))

#Step2 : Concatenating the features and training them
concat_train_x = np.concatenate((x1_train,x2_train,x3_train),axis=1)
concat_test_x = np.concatenate((x1_test,x2_test,x3_test),axis=1)
model_step2 = svm_train(y_train_flatten,concat_train_x, '-c 10 -t 0')

#Step 2 predict
p_label_step2, p_acc_step2, p_val_step2 = svm_predict(y_test_flatten,concat_test_x, model_step2)

#Accuracy step0 part 1
# print("Accuracy for Step 0 part 1 and feature vector X1:" + str(p_acc_step0_que1_case1))
# print("Accuracy for Step 0 part 1 and feature vector X2:" + str(p_acc_step0_que1_case2))
# print("Accuracy for Step 0 part 1 and feature vector X3:" + str(p_acc_step0_que1_case3))
#
# #Accuracy step0 part2
# print("Accuracy for Step 0 part 2 and feature vector X1:" + str(p_acc_step0_que2_case1))
# print("Accuracy for Step 0 part 2 and feature vector X2:" + str(p_acc_step0_que2_case2))
# print("Accuracy for Step 0 part 2 and feature vector X3:" + str(p_acc_step0_que2_case3))
#
# #Accuracy step1
# print("Accuracy for Step1: "+str((correct/len(predicted_class))*100))
#
# #Accuracy Step2
# print("Accuracy for Step2: "+str(p_acc_step2))
#
# print("----------------------------------------------------------------------------")

print("Accuracy for Step 0 part 1 and feature vector X1: " + str(p_acc_step0_que1_case1[0]))
print("Accuracy for Step 0 part 1 and feature vector X2: " + str(p_acc_step0_que1_case2[0]))
print("Accuracy for Step 0 part 1 and feature vector X3: " + str(p_acc_step0_que1_case3[0]))

print("Accuracy for Step 0 part 2 and feature vector X1: " + str(p_acc_step0_que2_case1[0]))
print("Accuracy for Step 0 part 2 and feature vector X2: " + str(p_acc_step0_que2_case2[0]))
print("Accuracy for Step 0 part 2 and feature vector X3: " + str(p_acc_step0_que2_case3[0]))

print("Accuracy for Step1: "+str((correct/len(predicted_class))*100))

print("Accuracy for Step2: "+str(p_acc_step2[0]))
