import sys
from PyQt5.QtWidgets import QMainWindow, QApplication
from PyQt5 import uic
from google.cloud import automl_v1beta1
from google.cloud import vision
import io
import cv2 

Ui_MainWindow, QtBaseClass = uic.loadUiType('C:\\Users\\Dennis\\Documents\\Horizon_Global\\Machine_Learning\\Implementation\\automlvision\\GoogleML.ui')
class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.pic_button.clicked.connect(self.getTakePicButtonResult)
        # images that are parameters of the window class
        self.loaded_image = ''
        self.westfalia_label_path = ''
        self.silver_label_path = ''
        self.id_label_path = ''
        self.starlock_1_path  = ''
        self.starlock_2_path = ''
        self.starlock_3_path = ''
        self.starlock_4_path = ''
        # project ID
        self.project_id_bike_carrier = "bike-carrier" 
        # model_IDs
        self.model_id_label = "IOD6317468357759074304" 
        self.model_id_starlocks = "IOD1549880386606071808"
        
    def getTakePicButtonResult(self):
        image = cv2.imread("C:\\Users\\Dennis\\Documents\\Horizon_Global\\Machine_Learning\\Implementation\\Trial images\\trial_cellphone_image (10).JPG")
        self.loaded_image = image
#        cv2.imshow("test", image)
#        cv2.waitKey(0)
        #cropping
        self.crop()
        #object detection
        self.detect_object(self.starlock_1_path, self.project_id_bike_carrier, self.model_id_starlocks)
        self.detect_object(self.starlock_2_path, self.project_id_bike_carrier, self.model_id_starlocks)
        self.detect_object(self.starlock_3_path, self.project_id_bike_carrier, self.model_id_starlocks)
        self.detect_object(self.starlock_4_path, self.project_id_bike_carrier, self.model_id_starlocks)
        self.detect_object(self.westfalia_label_path, self.project_id_bike_carrier, self.model_id_label)
        self.detect_object(self.silver_label_path, self.project_id_bike_carrier, self.model_id_label)
        self.detect_object(self.id_label_path, self.project_id_bike_carrier, self.model_id_label)
        self.detect_text('test_key.png', 'save_test.png')
        print('done')
        
        
    # cropping function
    def crop(self):
        #[x1,y1][x2,y2]
        #Westfalia ([934,2458],[1428,2559])
        #Silver ([1707, 1451],[1909,1680])
        #ID ([677, 648],[966, 770])
        
        #SL1 ([477,1167],[600,1428])
        #SL2 ([457,1891],[567,2160])
        #SL3 ([1774,1149],[1878,1410])
        #SL4 ([1786,1862],[1887,2129])
        
        # note: OpenCV coordinates are matrix based, meaning (row, column)
        # meaning (y, x) thus self.loaded_image[y1:y2, x1:x2] 
        # temp images are saved to be loaded as bytes later
        temp_image_names = ['temp_westfalia.PNG', 'temp_silver.PNG', 
                            'temp_id.PNG', 'temp_starlock1.PNG', 
                            'temp_starlock2.PNG', 'temp_starlock3.PNG',
                            'temp_starlock4.PNG']
        
        
        # setting the location of the images in the code
        # this will be done once hi res images become available
#        self.westfalia_label_path = temp_image_names[0]
#        self.silver_label_path = temp_image_names[1]
#        self.id_label_path = temp_image_names[2]
#        self.starlock_1_path = temp_image_names[3]
#        self.starlock_2_path = temp_image_names[4]
#        self.starlock_3_path = temp_image_names[5]
#        self.starlock_4_path = temp_image_names[6]
        
        # for testing
        self.westfalia_label_path = 'beta_westfalia.png'
        self.silver_label_path = 'beta_silver.png'
        self.id_label_path = 'beta_id.png'
        self.starlock_1_path = 'beta_star1.png'
        self.starlock_2_path = 'beta_star2.png'
        self.starlock_3_path = 'beta_star3.png'
        self.starlock_4_path = 'beta_star4.png'        
        
        
        # labels
        cv2.imwrite(temp_image_names[0], self.loaded_image[2458:2559, 934:1428])
        cv2.imwrite(temp_image_names[1], self.loaded_image[1451:1680, 1707:1909])
        cv2.imwrite(temp_image_names[2], self.loaded_image[648:770, 677:966])
        # starlocks
        cv2.imwrite(temp_image_names[3], self.loaded_image[1167:1428, 477:600])
        cv2.imwrite(temp_image_names[4], self.loaded_image[1891:2160, 457:567])
        cv2.imwrite(temp_image_names[5], self.loaded_image[1149:1410, 1774:1878])
        cv2.imwrite(temp_image_names[6], self.loaded_image[1862:2129, 1786:1887])
        
        
    # object detection and bounding box drawing function
    def detect_object(self, file_path, project_id, model_id):
        
        # reading byte file in
        with open(file_path, 'rb') as ff:
            content = ff.read()
        
        # reading drawing image in
        display_image = cv2.imread(file_path)
        # dimensions of image
        h, w, d = display_image.shape
        
        #identified number re-check counter
        label_count = 0
         #font for text 
        font = cv2.FONT_HERSHEY_SIMPLEX        
        #for labelling to count the number of valid boxes
        box_count = 0
        #to shift the label sufficiently
        text_shift = 10
        
        prediction_client = automl_v1beta1.PredictionServiceClient()
      
        name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
        payload = {'image': {'image_bytes': content }}
        params = {}
        request = prediction_client.predict(name, payload, params)
        
        # looping through returned values to draw bounding boxes
        # counter variables looping through returned values
        count = 0
        for next_payload in request.payload:
            
            # getting the x and y values for the bounding boxes
            x1 = int(request.payload[count].image_object_detection.bounding_box.normalized_vertices[0].x * w)
            y1 = int(request.payload[count].image_object_detection.bounding_box.normalized_vertices[0].y * h)
            x2 = int(request.payload[count].image_object_detection.bounding_box.normalized_vertices[1].x * w)
            y2 = int(request.payload[count].image_object_detection.bounding_box.normalized_vertices[1].y * h)
            identified_object = request.payload[count].display_name
            
            cv2.rectangle(display_image, (x1,y1), (x2, y2), (0,255,0), 3)
            cv2.putText(display_image, str(identified_object), (0,label_count+110), font, 5, (0,255,0), 2, cv2.LINE_AA)
            
            count+=1
        
        cv2.imwrite("testitest " + file_path, display_image)

        return request  # waits till request is returned
    

    # detecting text - with logic
    def detect_text(self, file_path, save_name):
        draw_image = cv2.imread(file_path)
        
        # reading byte file in
        with open(file_path, 'rb') as ff:
            content = ff.read()
        
        client = vision.ImageAnnotatorClient()

        image = vision.types.Image(content=content)
    
        response = client.text_detection(image=image)
        texts = response.text_annotations
                
        #font for text 
        font = cv2.FONT_HERSHEY_SIMPLEX        
        #for labelling to count the number of valid boxes
        box_count = 0
        #to shift the label sufficiently
        text_shift = 10
        
        #identified number re-check counter
        label_count = 0
       
        print('Texts:')
    
        for text in texts:
            print('\n"{}"'.format(text.description))
    
            vertices = (['({},{})'.format(vertex.x, vertex.y)
                        for vertex in text.bounding_poly.vertices])
            
            #for cross checking    
            x1, y1, x2, y2 = text.bounding_poly.vertices[0].x, text.bounding_poly.vertices[0].y, text.bounding_poly.vertices[2].x, text.bounding_poly.vertices[2].y
            x_text, y_text = text.bounding_poly.vertices[box_count].x, text.bounding_poly.vertices[box_count].y
            
            if (len(text.description) == 3):
            
                cv2.rectangle(draw_image, (x1,y1), (x2, y2), (0,255,0), 3)
                cv2.putText(draw_image, str(box_count+1), (int(x_text+text_shift),int(y_text+text_shift)), font, 2, (0,255,0), 2, cv2.LINE_AA)
                cv2.putText(draw_image, str(text.description), (0,label_count+110), font, 5, (0,255,0), 2, cv2.LINE_AA)
                box_count += 1
                label_count += 100
            
            print('bounds: {}'.format(','.join(vertices)))
            
        cv2.imwrite(save_name, draw_image) 
        
        
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())

#FUNCTIONS THAT MUST BE CREATED 
    
    #1. Cropping function, hard-coded bounds for each part based on image x
    #2. Check keys, check starlocks, check labels x
    #3. Drawing function to return the image, with its bounding boxes/numbers
    #3.1. This can be done within the detection function
    #4. Text ID function, capable of 'logicing' the text returned by Google
    
    #Malvin
    #5. Display function for all images 
    #6. Buttons for each image OK/NOK from operator
    #7. 'Saving' type function
    