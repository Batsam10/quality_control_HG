import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from google.cloud import automl_v1beta1

Ui_MainWindow, QtBaseClass = uic.loadUiType('C:\\Users\\Dennis\\Documents\\Horizon_Global\\Machine_Learning\\Implementation\\automlvision\\GoogleML.ui')
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pic_button.clicked.connect(self.getTakePicButtonResult)
        
    def getTakePicButtonResult(self):
        self.ui.results_window.setText("take nice photos dude!!!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
    
# OBJECT DETECTION
# 'content' is base-64-encoded image data.
def get_prediction(content, project_id, model_id, file_path):
    prediction_client = automl_v1beta1.PredictionServiceClient()
  
    name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
    payload = {'image': {'image_bytes': content }}
    params = {}
    request = prediction_client.predict(name, payload, params)
    return request  # waits till request is returned
    
    with open(file_path, 'rb') as ff:
        content = ff.read()
    
# RUN: Label object detection 
project_id_bike_carrier = "bike-carrier" 
# label model ID
model_id_label = "IOD6317468357759074304" 
model_id_starlocks = "IOD1549880386606071808"


#FUNCTIONS THAT MUST BE CREATED 
    
    #1. Cropping function, hard-coded bounds for each part based on image
    #2. Check keys, check starlocks, check labels
    #3. Drawing function to return the image, with its bounding boxes/numbers
    #4. Text ID function, capable of 'logicing' the text returned by Google
    
    #Malvin
    #5. Display function for all images 
    #6. Buttons for each image OK/NOK from operator
    #7. 'Saving' type function
    