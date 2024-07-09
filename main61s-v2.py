###Developp just by me Léandre NAUDIN


from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtWidgets import QDialog, QApplication, QFileDialog
from PySide6.QtGui import QPixmap
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtGui import QCursor



import sys
import os
import time
import numpy as np
import tensorflow as tf
import cv2
import subprocess
import glob

from utils import label_map_util
from utils import leandre_vizualization as le_util
sys.path.append("..")


class Worker(QObject):

    global direction
    global min_detection
    global name_sure
    global passage
    global debut_passage
    global fin_passage
    global app_data_folder
    finished = Signal()
    progress = Signal(int)
    progress_bar_update = Signal(int)
    etat_avancement_update = Signal(str)
    image_update = Signal(np.ndarray)
    lancement_update = Signal()
    lancement_update_true = Signal()


    def run (self):
        global direction
        start_time_ = time.time()
        print(min_detection)

        if direction == 'Aucun direction' or direction == None or direction =='':

            self.finished.emit()
        else:
            print(direction)
            dir_temp = direction
            path = direction
            dir_temp = dir_temp.split('/')
            fichier = dir_temp.pop(-1)
            dir_temp = '/'.join(dir_temp) + '/'
            fichier_noext = fichier.split('.')
            fichier_noext = fichier_noext[0]
            fichier_avi = app_data_folder +  fichier_noext + 'temp.avi'
            fichier_mp4 = app_data_folder + fichier_noext + 'temp.mp4'
            fichier_audio = app_data_folder + fichier_noext + 'temp.wav'
            fichier_final = dir_temp + fichier_noext +'_' + name_sure + '_blur.mp4'



            if os.path.exists(fichier_avi):
                os.remove(fichier_avi)

            if os.path.exists(fichier_audio):
                os.remove(fichier_audio)

            if os.path.exists(fichier_mp4):
                os.remove(fichier_mp4)

            if os.path.exists(fichier_final):
                self.etat_avancement_update.emit("La vidéo a déja été traité ")
                self.finished.emit()
            else:
                # Path to frozen detection graph. This is the actual model that is used for the object detection.

                self.lancement_update.emit()

                PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

                # List of the strings that is used to add correct label for each box.
                PATH_TO_LABELS = './protos/face_label_map.pbtxt'

                NUM_CLASSES = 2

                label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
                categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                            use_display_name=True)
                category_index = label_map_util.create_category_index(categories)

                self.progress_bar_update.emit(0)
                value_progress = 0

                def load_image_into_numpy_array(image):
                    (im_width, im_height) = image.size
                    return np.array(image.getdata()).reshape(
                        (im_height, im_width, 3)).astype(np.uint8)

                cap = cv2.VideoCapture(path)
                out = None

                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # out = cv2.VideoWriter(fichier_avi, 0, fps, (width, height))

                self.etat_avancement_update.emit("Traitement des images en cours ...")
                detection_graph = tf.Graph()
                with detection_graph.as_default():
                    od_graph_def = tf.compat.v1.GraphDef()
                    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                        serialized_graph = fid.read()
                        od_graph_def.ParseFromString(serialized_graph)
                        tf.import_graph_def(od_graph_def, name='')

                with detection_graph.as_default():
                    config = tf.compat.v1.ConfigProto()
                    config.gpu_options.allow_growth = True
                    with tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
                        frame_num = 0
                        while frame_num > -1:
                            frame_num += 1
                            ret, image = cap.read()
                            if ret == 0:
                                break

                            if out is None:
                                [h, w] = image.shape[:2]
                                print('On est passé par la')
                                out = cv2.VideoWriter(fichier_avi, 0, fps, (w, h))

                            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                            # the array based representation of the image will be used later in order to prepare the
                            # result image with boxes and labels on it.
                            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                            image_np_expanded = np.expand_dims(image_np, axis=0)
                            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                            # Each box represents a part of the image where a particular object was detected.
                            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                            # Each score represent how level of confidence for each of the objects.
                            # Score is shown on the result image, together with the class label.
                            scores = detection_graph.get_tensor_by_name('detection_scores:0')
                            classes = detection_graph.get_tensor_by_name('detection_classes:0')
                            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
                            # Actual detection.
                            old_value_progress = value_progress
                            value_progress = int(round(frame_num / total_frame * 10000))
                            if value_progress - old_value_progress >= 1:
                                self.progress_bar_update.emit(value_progress)
                            start_time = time.time()
                            (boxes, scores, classes, num_detections) = sess.run(
                                [boxes, scores, classes, num_detections],
                                feed_dict={image_tensor: image_np_expanded})
                            elapsed_time = time.time() - start_time

                            print('inference time cost: {}'.format(elapsed_time))
                            boxes = np.squeeze(boxes)
                            scores = np.squeeze(scores)
                            # Visualization of the results of a detection.
                            #print(np.squeeze(classes).astype(np.int32)[0])
                            if (not passage) or (debut_passage <= frame_num <= fin_passage) :


                                image = le_util.visualize_mask(image,  ##
                                                               boxes,
                                                               np.squeeze(classes).astype(np.int32),
                                                               scores,
                                                               category_index,
                                                               use_normalized_coordinates=True,
                                                               min_score_thresh=min_detection,
                                                               only_one_class = 1)


                            image = np.array(image)  ##
                            self.image_update.emit(image)
                            out.write(image)

                        cap.release()
                        try:
                            out.release()
                        except:
                            self.etat_avancement_update.emit('Erreur : format non valide')
                            self.finished.emit()

                        self.etat_avancement_update.emit("Compilation de la vidéo en cours ...")
                        self.progress_bar_update.emit(0)
                        command = 'ffmpeg-4.4-full_build\\bin\\ffmpeg.exe -i "{input}" -ab 160k -ac 2 -ar 44100 -vn "{output}"'.format(
                            input=path, output=fichier_audio)
                        subprocess.call(command, shell=True)
                        count_prog = 0
                        while count_prog < 3301:
                            self.progress_bar_update.emit(count_prog)
                            count_prog += 1
                        command = 'ffmpeg-4.4-full_build\\bin\\ffmpeg.exe -i "{input}" -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 "{output}"'.format(
                            input=fichier_avi, output=fichier_mp4)
                        subprocess.call(command, shell=True)
                        while count_prog < 6701:
                            self.progress_bar_update.emit(count_prog)
                            count_prog += 1

                        command = 'ffmpeg-4.4-full_build\\bin\\ffmpeg.exe -i "{input}" -i "{audio}" -map 0:v -map 1:a -c:v copy -shortest "{output}"'.format(
                            input=fichier_mp4, audio=fichier_audio, output=fichier_final)
                        try:
                            subprocess.call(command, shell=True)
                        except:
                            print("passe par erreur de son")
                            self.etat_avancement_update.emit("Erreur : format non valide")

                        while count_prog < 10001:
                            self.progress_bar_update.emit(count_prog)
                            count_prog += 1
                        print("Le temps total est de  : ", start_time_ - time.time())

                        self.etat_avancement_update.emit("Votre vidéo est prête !")
                        open = True
                        try:
                            os.remove(fichier_audio)
                        except:
                            self.etat_avancement_update.emit("Erreur : Format non valide")
                            open = False
                        try:
                            os.remove(fichier_mp4)
                        except:
                            self.etat_avancement_update.emit("Erreur : Format non valide")
                            open = False
                        try:
                            os.remove(fichier_avi)
                        except:
                            self.etat_avancement_update.emit("Erreur : Format non valide")
                            open = False

                if open:
                    subprocess.Popen('explorer "{}"'.format(fichier_final.replace('/','\\')))
                self.finished.emit()
        self.lancement_update_true.emit()

        '''list_parasite = glob.glob("./*.avi") + glob.glob("./*.mp4") + glob.glob("./*.wav")
        for element in list_parasite:
            os.remove(element)'''


class Ui_MainWindow(QDialog):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setFixedSize(1000, 725)
        MainWindow.setStyleSheet("#menubar {background-color : white;}"
                                 "#statusbar {background-color : #d2eae8;}")

        app_icon = QtGui.QIcon()
        app_icon.addFile('icon/16x16.png', QtCore.QSize(16, 16))
        app_icon.addFile('icon/24x24.png', QtCore.QSize(24, 24))
        app_icon.addFile('icon/32x32.png', QtCore.QSize(32, 32))
        app_icon.addFile('icon/48x48.png', QtCore.QSize(48, 48))
        app_icon.addFile('icon/256x256.png', QtCore.QSize(256, 256))
        app.setWindowIcon(app_icon)


        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.centralwidget.setStyleSheet("QPushButton#Lancement {background-color: qlineargradient(x1: 0.05, y1:0.05, x2:0.95, y2:0.95, stop : 0 #DCE35B,stop:0.51 #45B649 ,stop:1 #DCE35B )}" 
                                         "QPushButton#Lancement {text-align: center;color: white; border-radius: 20px; border : none; }" 
                                        "QPushButton#Lancement:hover { background-position: right center; color: #fff ;}"
                                         "QPushButton#Lancement:hover {background-color: qlineargradient(x1: 0.15, y1:0.15, x2:0.85, y2:0.85, stop : 0 #DCE35B ,stop:0.51 #45B649 ,stop:1 #DCE35B  )}"
                                         "QPushButton#Lancement:pressed {background-color : #41b034;}"
                                         
                                         "QWidget#parametre_qualite {background-color : #9188f7; border-radius : 30px;}"                                        
                                         "QWidget#zone_utile {background-color : #9188f7; border-radius : 30px;}"
                                         "QWidget#zone_4 {border-radius : 30px; background-color : #9188f7;}"
                                         "#image {background-color : #9188f7; border-radius : 30px;}"
                                         
                                         "QWidget #centralwidget {background-color : #d2eae8;}"
                                         
                                         "QWidget#zone_selection {border-radius : 20px; background-color : #574cd8;}" #f4fab7
                                         "QWidget#zone_radio {border-radius : 20px; background-color : #574cd8;}" #f4fab7 
                                         

                                         "#progressBar { border-radius : 10px; text-align: center; font-weight : bold; font-size : 18px; letter-spacing: 2.5px; border : none;}"
                                         "#progressBar:chunk { background-color : #6ac65f; margin-right : 0.5px; border-radius : 10px;}"

                                         "QPushButton#Browser {background-color: qlineargradient(x1: 0.05, y1:0.05, x2:0.95, y2:0.95, stop : 0 #4CA1AF,stop:0.51 #C4E0E5 ,stop:1 #4CA1AF )}"
                                         "QPushButton#Browser {border : 0px solid black; border-radius : 15px; }"
                                         "QPushButton#Browser:hover {background-color: qlineargradient(x1: 0.15, y1:0.15, x2:0.85, y2:0.85, stop : 0 #4CA1AF,stop:0.51 #C4E0E5 ,stop:1 #4CA1AF )}"
                                         "QPushButton#Browser:pressed {background-color: #59c29f;}"

                                         "QPushButton#going_folder {background-color: qlineargradient(x1: 0.05, y1:0.05, x2:0.95, y2:0.95, stop : 0 #4CA1AF,stop:0.51 #C4E0E5 ,stop:1 #4CA1AF )}"
                                         "QPushButton#going_folder {border : 0px solid black; border-radius : 15px; }"
                                         "QPushButton#going_folder:hover {background-color: qlineargradient(x1: 0.15, y1:0.15, x2:0.85, y2:0.85, stop : 0 #4CA1AF,stop:0.51 #C4E0E5 ,stop:1 #4CA1AF )}"
                                         "QPushButton#going_folder:pressed {background-color: #59c29f;}"
                                         
                                     
                                         "QLabel {color : white;}"
                                         
                                         "QRadioButton {color : white; text-align : center;}"
                                         "QRadioButton::indicator {border : None; background-color : white; border-radius : 6px;}"
                                         "QRadioButton::indicator::checked {background-color : #454649; border-radius : 6px;}"
                                         
                                        "QLabel#Etat_avancement {color : white; font-weight : bold;}"
                                         
                                         "QLbel#label {padding : 0px; margin : -10px;}"
                                         
                                         "QCheckBox {color : white; text-align : center; padding-left : 0px;}"
                                         "QCheckBox::indicator {border-radius : 5px; background-color : white;}"
                                         "QCheckBox::indicator::checked {background-color : #454649;}"

                                        "QSlider::groove:horizontal {  border: 1px solid  #bbb; background: white; height: 10px; border-radius: 4px;}"
                                
                                        "QSlider::sub-page:horizontal{ background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0  #66e, stop: 1 #bbf);"
                                         "background: qlineargradient(x1: 0, y1: 0.2, x2: 1, y2: 1,stop: 0  #bbf, stop: 1 #55f); border: 1px solid  #777; height: 10px; border-radius: 4px;}"                                
                                        
                                        "QSlider::add-page:horizontal {background:  #fff; border: 1px solid  #777; height: 10px;  border-radius: 4px;}"
                                
                                        "QSlider::handle:horizontal {background:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,stop: 0  #eee, stop:1 #ccc);border: 1px solid  #777;"
                                        "width: 13px;margin-top: -2px; margin-bottom: -2px;   border-radius: 4px;}"
                                
                                        "QSlider::handle:horizontal:hover{background:qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0  #fff, stop:1 #ddd); border: 1px solid  #444; border-radius: 4px;}"
                                
                                        "QSlider::sub-page:horizontal:disabled{background: #bbb; border-color:  #999; }"
                                
                                        "QSlider::add-page:horizontal:disabled { background:  #eee;border-color:  #999; }"
                                
                                        "QSlider::handle:horizontal:disabled {  background:  #eee; border: 1px   solid  #aaa;   border-radius: 4px;}"

                                         )

        self.image = QtWidgets.QLabel(self.centralwidget)
        self.image.setGeometry(QtCore.QRect(30, 20, 640, 360))
        self.image.setText("")
        self.image.setObjectName("image")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.parametre_qualite = QtWidgets.QWidget(self.centralwidget)
        self.parametre_qualite.setGeometry(QtCore.QRect(690, 20, 291, 360))
        self.parametre_qualite.setStyleSheet("")
        self.parametre_qualite.setObjectName("parametre_qualite")

        self.zone_radio = QtWidgets.QWidget(self.parametre_qualite)
        self.zone_radio.setGeometry(QtCore.QRect(10, 10, 271, 181))
        self.zone_radio.setStyleSheet("")
        self.zone_radio.setObjectName("zone_radio")

        self.normal = QtWidgets.QRadioButton(self.zone_radio)
        self.normal.setGeometry(QtCore.QRect(80, 110, 101, 21))
        self.normal.setChecked(True)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.normal.setFont(font)
        self.normal.setObjectName("normal")
        self.sur = QtWidgets.QRadioButton(self.zone_radio)
        self.sur.setGeometry(QtCore.QRect(80, 145, 111, 21))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.sur.setFont(font)
        self.sur.setObjectName("sur")
        self.risque = QtWidgets.QRadioButton(self.zone_radio)
        self.risque.setGeometry(QtCore.QRect(80, 70, 101, 31))

        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.risque.setFont(font)
        self.risque.setObjectName("risque")



        self.info_detection = QtWidgets.QLabel(self.zone_radio)
        self.info_detection.setGeometry(QtCore.QRect(25, 5, 221, 61))
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        font.setBold(True)

        self.info_detection.setFont(font)
        self.info_detection.setObjectName("info_detection")
        self.info_detection.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zone_selection = QtWidgets.QWidget(self.parametre_qualite)
        self.zone_selection.setGeometry(QtCore.QRect(10, 200, 271, 151))
        self.zone_selection.setObjectName("zone_selection")
        self.label_fich = QtWidgets.QLabel(self.zone_selection)
        self.label_fich.setGeometry(QtCore.QRect(10, 30, 251, 31))
        font = QtGui.QFont()
        ##font.setFamily("Noto Sans Hebrew")
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.label_fich.setFont(font)
        self.label_fich.setStyleSheet("text-align : center;")
        self.label_fich.setObjectName("label_fich")
        self.label_fich.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label = QtWidgets.QLabel(self.zone_selection)
        self.label.setGeometry(QtCore.QRect(10, 85, 251, 31))
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(13)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.zone_utile = QtWidgets.QWidget(self.centralwidget)
        self.zone_utile.setGeometry(QtCore.QRect(30, 390, 641, 311))
        self.zone_utile.setObjectName("zone_utile")
        self.going_folder = QtWidgets.QPushButton(self.zone_utile)
        self.going_folder.setGeometry(QtCore.QRect(40, 230, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Light")
        font.setPointSize(13)
        font.setBold(True)
        #font.setWeight(75)
        self.going_folder.setFont(font)
        self.going_folder.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.going_folder.setObjectName("going_folder")
        self.going_folder.clicked.connect(self.click_on_going_foler)

        self.Browser = QtWidgets.QPushButton(self.zone_utile)
        self.Browser.setGeometry(QtCore.QRect(440, 230, 171, 31))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Light")
        font.setPointSize(13)
        font.setBold(True)
        #font.setWeight(75)
        self.Browser.setFont(font)
        self.Browser.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.Browser.setObjectName("Browser")
        self.Browser.clicked.connect(self.on_click_browse)


        self.progressBar = QtWidgets.QProgressBar(self.zone_utile)
        self.progressBar.setGeometry(QtCore.QRect(40, 30, 581, 23))
        #self.progressBar.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.WaitCursor))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.progressBar.setMinimum(0)
        self.progressBar.setMaximum(10000)


        self.Lancement = QtWidgets.QPushButton(self.zone_utile)
        self.Lancement.setGeometry(QtCore.QRect(130, 130, 391, 61))
        font = QtGui.QFont()
        font.setFamily("Orbitron")
        font.setPointSize(16)
        self.Lancement.setFont(font)
        self.Lancement.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
        self.Lancement.setObjectName("Lancement")
        self.Lancement.clicked.connect(self.main)
        self.Lancement.setEnabled(True)




        self.Etat_avancement = QtWidgets.QLabel(self.zone_utile)
        self.Etat_avancement.setGeometry(QtCore.QRect(40, 69, 581, 41))
        font = QtGui.QFont()
        font.setFamily("Noto Sans Light")
        font.setPointSize(13)
        font.setBold(True)
        #font.setWeight(75)
        self.Etat_avancement.setFont(font)
        self.Etat_avancement.setObjectName("Etat_avancement")
        self.Etat_avancement.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.zone_4 = QtWidgets.QWidget(self.centralwidget)
        self.zone_4.setGeometry(QtCore.QRect(690, 390, 291, 311))
        self.zone_4.setObjectName("zone_4")

        self.valide = QtWidgets.QCheckBox(self.zone_4)
        self.valide.setChecked(False)
        self.valide.setGeometry(30,10,241,61)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.valide.setFont(font)
        self.valide.setEnabled(False)
        self.valide.stateChanged.connect(self.active_passage)

        self.visualize = QtWidgets.QCheckBox(self.zone_4)
        self.visualize.setGeometry(30,80,241,61)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.visualize.setFont(font)
        self.visualize.setEnabled(False)


        '''
        self.debut_sel = QtWidgets.QSpinBox(self.zone_4)
        self.debut_sel.setGeometry(20,210, 91,41)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(11)
        self.debut_sel.setFont(font)
        self.debut_sel.setMinimum(0)
        self.debut_sel.setMaximum(0)
        self.debut_sel.setEnabled(False)
        self.debut_sel.valueChanged.connect(self.change_value_debut)


        self.fin_sel = QtWidgets.QSpinBox(self.zone_4)
        self.fin_sel.setGeometry(150, 210, 91, 41)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(11)
        self.fin_sel.setFont(font)
        self.fin_sel.setMinimum(0)
        self.fin_sel.setMaximum(1)
        self.fin_sel.setValue(1)
        self.fin_sel.setEnabled(False)'''

        self.debut_sel = QtWidgets.QSlider(Qt.Orientation.Horizontal, self.zone_4)
        self.debut_sel.setGeometry(20, 150, 180, 40)
        self.debut_sel.setMinimum(0)
        self.debut_sel.setMaximum(0)
        self.debut_sel.setValue(0)
        self.debut_sel.setEnabled(False)
        self.debut_sel.setSingleStep(1)
        self.debut_sel.valueChanged.connect(self.change_value_debut)

        self.fin_sel = QtWidgets.QSlider(Qt.Orientation.Horizontal, self.zone_4)
        self.fin_sel.setGeometry(20, 200, 180, 40)
        self.fin_sel.setMinimum(0)
        self.fin_sel.setMaximum(0)
        self.fin_sel.setValue(0)
        self.fin_sel.setEnabled(False)
        self.fin_sel.setSingleStep(1)
        self.fin_sel.valueChanged.connect(self.change_value_fin)

        self.debut_sel_label = QtWidgets.QLabel(self.zone_4)
        self.debut_sel_label.setGeometry(210,150, 60, 40)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.debut_sel_label.setFont(font)

        self.fin_sel_label = QtWidgets.QLabel(self.zone_4)
        self.fin_sel_label.setGeometry(210, 200, 60, 40)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setBold(True)
        font.setPointSize(13)
        self.fin_sel_label.setFont(font)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1000, 21))
        self.menubar.setObjectName("menubar")
        self.menuChercher_un_fichier = QtWidgets.QMenu(self.menubar)
        self.menuChercher_un_fichier.setObjectName("menuChercher_un_fichier")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionOuvrir_le_dernier_dossier_trait = QtGui.QAction(MainWindow)
        self.actionOuvrir_le_dernier_dossier_trait.setObjectName("actionOuvrir_le_dernier_dossier_trait")
        self.actionChercher_un_fichier = QtGui.QAction(MainWindow)
        self.actionChercher_un_fichier.setObjectName("actionChercher_un_fichier")
        self.actionLancer_le_floutage = QtGui.QAction(MainWindow)
        self.actionLancer_le_floutage.setObjectName("actionLancer_le_floutage")
        self.menuChercher_un_fichier.addAction(self.actionOuvrir_le_dernier_dossier_trait)
        self.menuChercher_un_fichier.addAction(self.actionChercher_un_fichier)
        self.menuChercher_un_fichier.addAction(self.actionLancer_le_floutage)
        self.menubar.addAction(self.menuChercher_un_fichier.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.risque, self.normal)
        MainWindow.setTabOrder(self.normal, self.sur)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Flouteur - lina-software.com"))
        self.normal.setText(_translate("MainWindow", "Moyen"))
        self.sur.setText(_translate("MainWindow", "Élevé"))
        self.risque.setText(_translate("MainWindow", "Faible"))
        self.info_detection.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:13pt;\">Choisisez le seuil de<br/>détection des visages</span></p></body></html>"))
        self.label_fich.setText(_translate("MainWindow", "Fichier sélectionnée :"))
        self.label.setText(_translate("MainWindow", "Aucun fichier sélectionnée"))
        self.going_folder.setText(_translate("MainWindow", "Dernier fichier"))
        self.Browser.setText(_translate("MainWindow", "Chercher fichier"))
        self.Lancement.setText(_translate("MainWindow", "LANCEMENT DU FLOUTAGE"))
        self.Etat_avancement.setText(_translate("MainWindow", "En attente du lancement du floutage ..."))
        self.valide.setText(_translate("MainWindow", "    Activer la sélection d'un \n    passage précis à flouter"))
        '''self.menuChercher_un_fichier.setTitle(_translate("MainWindow", "Menu"))
        self.actionOuvrir_le_dernier_dossier_trait.setText(_translate("MainWindow", "Ouvrir le dernier dossier traité"))
        self.actionChercher_un_fichier.setText(_translate("MainWindow", "Chercher un fichier"))
        self.actionLancer_le_floutage.setText(_translate("MainWindow", "Lancer le floutage"))'''
        self.visualize.setText(_translate("MainWindow", "    Afficher le début et \n    la fin de la sélection"))
        self.debut_sel_label.setText(_translate("MainWindow","Début"))
        self.fin_sel_label.setText(_translate("MainWindow","Fin"))


    def on_click_browse(self):
        global direction
        temp = direction
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'C:', '(*.mp4);;')
        direction = fname[0]
        if direction == "":
            direction = temp
        else:
            self.valide.setEnabled(True)
            self.active_passage()
            fichier = direction.split('/')
            fichier = fichier[-1]
            self.label.setText(fichier)
            self.debut_sel.setValue(0)
            self.fin_sel.setValue(1)

    def click_on_going_foler(self):
        if not direction == 'Aucun direction':
            temp_dir = direction.split('/')
            temp_dir.pop(-1)
            temp_dir = '\\'.join(temp_dir)
            subprocess.Popen('explorer "{}"'.format(temp_dir))

    def main(self):
        global name_sure
        global min_detection
        global passage
        global debut_passage
        global fin_passage
        global app_data_folder
        min_detection, name_sure = self.get_radio_check()
        passage = self.valide.isChecked()
        debut_passage = self.debut_sel.value()
        fin_passage = self.fin_sel.value()
        self.thread = QThread()
        self.worker = Worker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.worker.progress_bar_update.connect(self.change_value_progress_bar)
        self.worker.etat_avancement_update.connect(self.change_value_etat_avancement)
        self.worker.image_update.connect(self.change_label_image)
        self.worker.lancement_update.connect(self.update_lancement)
        self.worker.lancement_update_true.connect(self.update_lancement_true)
        self.thread.start()

    def change_value_progress_bar (self, value):
        self.progressBar.setValue(value)

    def change_value_etat_avancement(self, value):
        _translate = QtCore.QCoreApplication.translate
        html = "<html><head/><body><p><span style=\" font-size:13pt;\">{}</span></p></body></html>".format(value)
        self.Etat_avancement.setText(_translate("MainWindow", html))

    def get_radio_check(self):
        if self.risque.isChecked():
            return 0.2, "faible"
        elif self.normal.isChecked():
            return 0.3, "moyen"
        elif self.sur.isChecked():
            return 0.45, "élevé"

    def change_label_image(self,  opencv_img):
        rgbImage = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                         QtGui.QImage.Format.Format_RGB888)
        convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
        pixmap = QPixmap(convertToQtFormat)
        resizeImage = pixmap.scaled(640, 360, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        self.image.setPixmap(resizeImage)

    def update_lancement(self):

        self.Lancement.setEnabled(False)
        self.Lancement.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ForbiddenCursor))

    def update_lancement_true(self):

        self.Lancement.setEnabled(True)
        self.Lancement.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))

    def active_passage(self):

        cap = cv2.VideoCapture(direction)
        total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if self.valide.isChecked():
            self.visualize.setEnabled(True)
            cap = cv2.VideoCapture(direction)
            total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.debut_sel.setEnabled(True)
            self.fin_sel.setEnabled(True)
            self.debut_sel.setMaximum(total_frame - 1)
            self.fin_sel.setMaximum(total_frame - 1)
        else:
            self.debut_sel.setValue(0)
            self.fin_sel.setValue(1)
            self.visualize.setChecked(False)
            self.visualize.setEnabled(False)
            self.debut_sel.setEnabled(False)
            self.fin_sel.setEnabled(False)


    def change_value_debut(self):
        if self.debut_sel.value() >= self.fin_sel.value():
            self.debut_sel.setValue(self.fin_sel.value() -1)

        if self.visualize.isChecked():

            cap = cv2.VideoCapture(direction)
            cap.set(1, self.debut_sel.value())
            ret, frame = cap.read()
            try:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                                 QtGui.QImage.Format.Format_RGB888)
                convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
                pixmap = QPixmap(convertToQtFormat)
                resizeImage = pixmap.scaled(640, 360, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.image.setPixmap(resizeImage)
            except:
                pass

    def change_value_fin(self):

        if self.visualize.isChecked():

            cap = cv2.VideoCapture(direction)
            cap.set(1, self.fin_sel.value())
            ret, frame = cap.read()
            try:
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0],
                                                 QtGui.QImage.Format.Format_RGB888)
                convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
                pixmap = QPixmap(convertToQtFormat)
                resizeImage = pixmap.scaled(640, 360, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
                self.image.setPixmap(resizeImage)
            except:
                pass




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    global direction
    global min_detection
    global name_sure
    global passage
    global debut_passage
    global fin_passage
    global app_data_folder
    app_data_folder = os.environ['ALLUSERSPROFILE']
    passage = False
    debut_passage = 0
    fin_passage = 1
    name_sure = "normal"
    min_detection = 0.4
    direction = 'Aucun direction'
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())