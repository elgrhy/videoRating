import sys

import cv2
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QPalette, QPixmap
from PyQt5.QtWidgets import (QApplication, QFileDialog, QLabel, QPushButton,
                             QWidget)
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


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
        self.initUI()

    def initUI(self):
        self.set_dark_mode_style()

        self.program_name_label = QLabel("Video prediction pro", self)
        self.program_name_label.setGeometry(10, 10, 300, 50)
        font = QFont()
        font.setPointSize(18)
        font.setBold(True)
        self.program_name_label.setFont(font)

        # Define the GUI elements
        self.video_label = QLabel(self)
        self.video_label.setText("No video selected")
        self.video_label.setGeometry(10, 70, 300, 20)
        self.video_label.setStyleSheet('color: white; font-size: 14pt; font-family: Arial;')

        self.select_button = QPushButton("Select video", self)
        self.select_button.setGeometry(200, 150, 100, 30)
        self.select_button.setStyleSheet('background-color: #4CAF50; color: white; font-size: 14pt; font-family: Arial; border: none; border-radius: 5px; padding: 5px;')
        self.select_button.clicked.connect(self.select_video)

        self.analyze_button = QPushButton("Analyze video", self)
        self.analyze_button.setGeometry(420, 150, 100, 30)
        self.analyze_button.setStyleSheet('background-color: #4CAF50; color: white; font-size: 14pt; font-family: Arial; border: none; border-radius: 5px; padding: 5px;')
        self.analyze_button.clicked.connect(self.analyze_video)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(10, 180, 400, 20)
        self.result_label.setStyleSheet('color: white; font-size: 14pt; font-family: Arial;')

        # Define the window properties
        self.setGeometry(100, 100, 800, 400)
        self.setWindowTitle("Video prediction pro")
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

    def select_video(self):
        # Show a file dialog to choose a video file
        filename, _ = QFileDialog.getOpenFileName(self, "Select video file", "", "Video files (*.mp4 *.avi)")
        if filename:
            self.video_path = filename
            self.video_label.setText(filename)

    def analyze_video(self):
        if self.video_path is None:
            self.result_label.setText("Please select a video file")
            self.result_label.setGeometry(10, 180, 400, 20)
            self.result_label.setStyleSheet("color: red")
            return
        # Extract features from the selected video
        features = extract_features(self.video_path)
        # Make a prediction using the trained model
        prediction = model.predict([features.astype(float)])[0]
        # Display the prediction to the user
        self.result_label.setText("Prediction: {:.2f}% likelihood of being popular".format(prediction * 100))
        self.result_label.setGeometry(10, 180, 400, 20)
        self.result_label.setStyleSheet("color: white") 
        return

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = VideoAnalyzer()
    sys.exit(app.exec_())
