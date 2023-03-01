import sys

import cv2
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QCursor, QFont, QPalette, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QFormLayout,
                             QGridLayout, QHBoxLayout, QLabel, QPushButton,
                             QSizePolicy, QSpacerItem, QVBoxLayout, QWidget)
from sklearn.linear_model import LinearRegression


# Define the features to extract from the video
def extract_features(video_path):
    cap = cv2.VideoCapture(video_path)
    duration = cap.get(cv2.CAP_PROP_POS_MSEC)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    bitrate = cap.get(cv2.CAP_PROP_BITRATE)
    cap.release()
    return np.array([duration, width, height, fps, bitrate])


# Load dataset from file
dataset = np.genfromtxt('/Users/elgrhy/developer/videoRating/env/videoRating/video_dataset.csv', delimiter=',', dtype=None, names=True, encoding=None)

# Convert structured array to regular numpy array
X = structured_to_unstructured(dataset[['frame_count', 'width', 'height', 'fps', 'bitrate']]).astype(np.float64)

# Extract labels
y = np.zeros(len(dataset))
y[dataset['label'] == 'Real'] = 1.0



# Train a linear regression model on the dataset
model = LinearRegression()
model.fit(X, y)

# Define the GUI window
class VideoAnalyzer(QWidget):
    def __init__(self):
        super().__init__()

        self.video_path = None
        self.analysis_in_progress = False
        self.initUI()

    def initUI(self):
        self.set_dark_mode_style()

        # Define the GUI elements
        self.video_label = QLabel(self)
        self.video_label.setText("No video selected")
        self.video_label.setStyleSheet('color: #FFF; font-size: 16pt; font-family: Arial;')
        self.video_label.setAlignment(Qt.AlignCenter)

        self.select_button = QPushButton("Select video", self)
        self.select_button.setStyleSheet("QPushButton { border: 2px solid #BC006C; color: #FFF; font-size: 14pt; font-family: 'shanti'; border-radius: 25px; padding: 15px 30px; } QPushButton:hover { background: #BC006C; }")
        self.select_button.clicked.connect(self.select_video)

        self.analyze_button = QPushButton("Analyze video", self)
        self.analyze_button.setStyleSheet("QPushButton { border: 2px solid #BC006C; color: #FFF; font-size: 14pt; font-family: 'shanti'; border-radius: 25px; padding: 15px 30px; } QPushButton:hover { background: #BC006C; }")
        self.analyze_button.clicked.connect(self.analyze_video)

        self.result_label = QLabel(self)
        self.result_label.setStyleSheet('color: #FFF; font-size: 16pt; font-family: Arial;')
        self.result_label.setAlignment(Qt.AlignCenter)
        #self.result_label.setText(self.display_recommendation)

        self.recommendation_label = QLabel(self)
        self.recommendation_label.setStyleSheet('color: #FFF; font-size: 14pt; font-family: Arial;')
        self.recommendation_label.setAlignment(Qt.AlignCenter)



        self.logo_label = QLabel(self)
        self.logo_pixmap = QPixmap('/Users/elgrhy/developer/videoRating/env/videoRating/20230301_201613_0000-removebg-preview.png')
        self.logo_label.setPixmap(self.logo_pixmap.scaled(200, 100, Qt.KeepAspectRatio))
        self.logo_label.setAlignment(Qt.AlignCenter)

        self.note_label = QLabel(self)
        self.note_label.setText("Note: This program uses a machine learning model to predict the likelihood of a video being popular.")
        self.note_label.setStyleSheet('color: #FFF; font-size: 14pt; font-family: Arial;')
        self.note_label.setAlignment(Qt.AlignCenter)

        # Define the window layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.addWidget(self.logo_label)
        self.form_layout = QFormLayout()
        self.form_layout.addRow(self.video_label)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.analyze_button)
        self.form_layout.addRow(button_layout)
        spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.form_layout.addItem(spacer)
        self.form_layout.addRow(self.result_label)
        self.form_layout.setVerticalSpacing(30)
        self.form_layout.setContentsMargins(50, 50, 50, 50)
        self.main_layout.addLayout(self.form_layout)
        self.main_layout.addWidget(self.note_label)
        
        # Define the window properties
        self.setGeometry(100, 100, 800, 500)
        self.setWindowTitle("Video prediction pro")
        self.set_dark_mode_style()
        self.show()

    def set_dark_mode_style(self):
        # Set the dark mode palette
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(45, 45, 45))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Base, QColor(18, 18, 18))
        palette.setColor(QPalette.AlternateBase, QColor(45, 45, 45))
        palette.setColor(QPalette.ToolTipBase, Qt.white)
        palette.setColor(QPalette.ToolTipText, Qt.white)
        palette.setColor(QPalette.Text, Qt.white)
        palette.setColor(QPalette.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ButtonText, Qt.white)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42,130, 218))
        palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.HighlightedText, Qt.black)
        self.setPalette(palette)

        font = QFont()
        font.setFamily('Arial')
        font.setPointSize(14)
        self.setFont(font)

    def display_recommendation(self, percentage):
        if percentage < 50:
            message = "Your video needs improvement in the following areas:\n\n"
            message += "- Increase video resolution\n"
            message += "- Increase video duration\n"
            message += "- Increase video frame rate\n"
        else:
            message = "Your video is good, but there's still room for improvement:\n\n"
            message += "- Increase video duration\n"
            message += "- Improve video content\n"
            message += "- Add background music\n"

    def select_video(self):
        # Show a file dialog to choose a video file
        filename, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Video files (*.mp4 *.avi)")
        if filename:
            self.video_path = filename
            self.video_label.setText(filename)

    def analyze_video(self):
        if self.video_path is None:
            self.result_label.setText("Please select a video file")
            self.result_label.setGeometry(10, 220, 800, 50)
            self.result_label.setStyleSheet('color: red; font-size: 16pt; font-family: Arial; padding: 75px')
            return
        # Extract features from the selected video
        features = extract_features(self.video_path)
        # Make a prediction using the trained model
        prediction = model.predict([features.astype(float)])[0]
        # Display the prediction to the user
        self.result_label.setText("Prediction: {:.2f}% likelihood of being popular".format(prediction * 100))
        self.result_label.setGeometry(10, 220, 800, 50)
        self.result_label.setStyleSheet('color: white; font-size: 16pt; font-family: Arial; padding: 75px') 
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoAnalyzer()
    sys.exit(app.exec_())
