'''
Created on Dec 5, 2015

@author: Joe
'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import matplotlib as plt
import timeit
import time
from matplotlib.pyplot import savefig

#high frequency model to predict appliance based on HF field
def model(freqData, target):
    start = timeit.default_timer()
    neighbors = 15
    model = KNeighborsClassifier(neighbors)
    f = open('data/H3/Results/Runtime Results.txt', 'a')
    f.write("\n\nHouse 3 Results -- Program start at "+time.strftime("%Y-%M-%D %H:%M:%S"))
    
    X_train, X_test, Y_train, Y_test = train_test_split(freqData, target, train_size=.80)
    split = timeit.default_timer()
    f.write("\nSplit in "+ str(split-start)+" seconds.")
    
    model.fit(X_train, Y_train) 
    fit = timeit.default_timer()
    f.write("\nFit in "+ str(fit-split)+" seconds.")
    
    prediction = model.predict(X_test)
    Y_test_probability = model.predict_proba(X_test)
    predict = timeit.default_timer()
    f.write("\nPredict in "+ str(predict-fit)+" seconds.")
    
    f.write("\nAccuracy on test = %.5f" % metrics.accuracy_score(prediction, Y_test))
    f.write("\nHamming loss on test = %.5f" % metrics.hamming_loss(prediction, Y_test))

    # Use the metrics.roc_curve function to get the true positive rate (tpr) and false positive rate (fpr)
    print(Y_test)
    print (Y_test_probability)
    fpr, tpr, thresholds = metrics.roc_curve(Y_test, Y_test_probability)
    
    # Get the area under the curve (AUC)
    acc = np.mean(cross_validation.cross_val_score(model, freqData, target, scoring="accuracy"))
    auc = np.mean(cross_validation.cross_val_score(model, freqData, target, scoring="roc_auc"))
    
        # Plot the ROC curve
    plt.plot(fpr, tpr, label="KNN (AUC = " + str(round(auc, 2)) + ")")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot([0,1], [0,1], 'k--', label="Random")
    plt.title('Receiver Operating Characteristics')
    plt.legend(loc=2)
    savefig("IDS "+time.strftime("%Y-%M-%D %H:%M:%S")+".png", bbox_inches='tight')
    
    stop = timeit.default_timer()
    f.write("\nTotal runtime: "+str(stop-start)+" seconds.")
    print("Routine complete.")
    
    # output is now written to a file