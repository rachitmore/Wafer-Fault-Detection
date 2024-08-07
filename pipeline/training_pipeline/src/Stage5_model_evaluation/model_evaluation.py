import datetime,sys
import pandas as pd
import numpy as np
import contextlib
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score
from sklearn.metrics import roc_curve,RocCurveDisplay,PrecisionRecallDisplay
from Utils.utils import FileOperation
from app_tracking.logger import App_Logger
from app_tracking.exception import AppException


class ModelEvaluation:
        def __init__(self, current_time):
                try:
                        print("Model evaluation has been started ..")
                        self.current_time = current_time
                        self.filepath = f"artifacts/logs/Stage5_ModelEvaluation/{self.current_time}.txt"
                        self.logging = App_Logger(self.filepath)
                        self.fileoperation = FileOperation()
                        self.model1 = self.fileoperation.load_model("artifacts/models/TrainAndTest/GradientBoostingClassifier.pkl")
                        self.model2 = self.fileoperation.load_model("artifacts/models/TrainAndTest/HistGradientBoostingClassifier.pkl")
                        self.model3 = self.fileoperation.load_model("artifacts/models/TrainAndTest/XGBClassifier.pkl")
                        self.model4 = self.fileoperation.load_model("artifacts/models/TrainAndTest/LGBMClassifier.pkl")
                        self.model5 = self.fileoperation.load_model("artifacts/models/TrainAndTest/CatBoostClassifier.pkl")
                        self.models = [self.model1, self.model2, self.model3, self.model4, self.model5]
                        self.X_train = load_npz('artifacts/data/X_trainSparse_matrix.npz').toarray()
                        self.X_test = load_npz('artifacts/data/X_testSparse_matrix.npz').toarray()
                        self.y_train = pd.read_csv("artifacts/data/y_train.csv")
                        self.y_test = pd.read_csv(r"artifacts/data/y_test.csv")
                        self.targets = ["0" , "1"]
                        with contextlib.suppress():
                                self.model_evaluation()
                                
                except Exception as e:
                        self.logging.log(str(e))
                        raise AppException(e,sys) from e


        def model_evaluation(self):
                for model in self.models:
                        try:
                                self.logging.log("Model Evaluation started")
                                model_name = type(model).__name__
                                self.logging.log(f"Model - {model_name}")
                                self.logging.log("Calculating training accuracy")
                                training_accuracy_score = model.score(self.X_train,self.y_train)# Calculate accuracy
                                self.logging.log(f"Training accuracy has been calculated {training_accuracy_score}")
                        
                                self.logging.log("Predicting X_test..")
                                self.y_pred = model.predict(self.X_test)
                                self.y_predDf = pd.DataFrame(self.y_pred)
                                self.fileoperation.save_data_to_csv(self.y_predDf,"artifacts/data/y_pred.csv")
                                self.logging.log("X_test has been predicted and saved to  successfully")

                                self.logging.log("creating classification report..")
                                self.cr = classification_report(self.y_test, self.y_pred,target_names = self.targets)    
                                self.logging.log(f"Classification Report -\n{self.cr}")

                                self.logging.log("creating confusion_matrix..")
                                self.cm = confusion_matrix(self.y_test, self.y_pred)
                                self.logging.log(f"Confusion matirx -\n{self.cm}")
                                
                                self.logging.log("Calculating Auc and Roc score")
                                y_pred_probs = model.predict_proba(self.X_test)
                                self.auc_roc = roc_auc_score(self.y_test,y_pred_probs[:, 1])
                                self.logging.log(f"AUC-ROC Score - {self.auc_roc}")
                                
                                self.logging.log("Plotting ROC and PR curve")
                                self.plot_roc_and_pr_curve(model,model_name)
                                self.logging.log("Model Evaluation has been done successfully")
                                
                        except Exception as e:
                                self.logging.log(str(e))
                                raise AppException(e,sys) from e
      
        def plot_roc_and_pr_curve(self,model,model_name):
                try:
                        self.logging.log("Creating RocCurveDisplay")
                        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred, pos_label = model.classes_[1])
                        roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)#Create RocCurveDisplay

                        self.logging.log("Creating PrecisionRecallDisplay")
                        pr_display = PrecisionRecallDisplay.from_predictions(self.y_test, self.y_pred)# Create PrecisionRecallDisplay

                        self.logging.log("Combining the display objects into a single plot")
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))# 

                        roc_display.plot(ax=ax1,)
                        pr_display.plot(ax=ax2,)

                        # Add titles and labels
                        ax1.set_title('ROC Curve')
                        ax1.set_xlabel('False Positive Rate (FPR)')
                        ax1.set_ylabel('True Positive Rate (TPR)')

                        ax2.set_title('Precision-Recall Curve')
                        ax2.set_xlabel('Recall')
                        ax2.set_ylabel('Precision')

                        plt.savefig(f"artifacts/plots/{model_name}_roc_and_pr.png")
                        self.logging.log(f"Plot has been saved to artifacts/plots/{model_name}_roc_and_pr.png successfully")
                except Exception as e:
                        self.logging.log(str(e))
                        raise AppException(e,sys) from e



        
     