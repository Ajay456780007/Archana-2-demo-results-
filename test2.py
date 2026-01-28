from Analysis import ANFIS_LGBM_Model
import numpy as np
from Analysis import incremental_update
from sklearn.model_selection import train_test_split
mm = np.load("New Dataset/Physionet/n_Features.npy")
label = np.load("New Dataset/Physionet/n_Labels.npy")

x_train,x_test,y_train,y_test = train_test_split(mm,label,test_size=0.25,random_state=42)

metrics = ANFIS_LGBM_Model(x_train,x_test,y_train,y_test,10)
print("Completed First Run.....")
metrics1 = ANFIS_LGBM_Model(x_train,x_test,y_train,y_test,10)
print("Completed Second Run...")

# incremental_update(Anfis_model,x_train,x_test,epochs=100)
