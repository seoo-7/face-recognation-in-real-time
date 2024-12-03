import face_recognition
import cv2
import numpy as np
import os

class SimpleFacerec:
    def __init__(self):
        # A list that stores the face encodings of known individuals
        self.known_face_encodings = [] 
        #  A list that stores the names corresponding to the "known_face_encodings"
        self.known_face_names = []
        self.frame_resizing = 0.25 #improve performance by reducing the resolution of input frames
        

    # This method loads face encodings and names from image files in a specified directory
    def load_encoding_images(self, images_path):
   
        for img_path in os.listdir(images_path):
            img = cv2.imread(os.path.join(images_path, img_path))
            img_encoding = face_recognition.face_encodings(img)[0] # this creates a numerical representation of the face
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(os.path.splitext(img_path)[0])


    # This method detects and identifies faces in a given frame.
    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # this method detects face locations in the resized frame
        face_locations = face_recognition.face_locations(rgb_small_frame) 
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)#method generates encodings for detected faces

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Find the closest known face encoding
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale back up face locations to match the original frame size
        face_locations = [(int(top / self.frame_resizing), int(right / self.frame_resizing), 
                           int(bottom / self.frame_resizing), int(left / self.frame_resizing)) 
                          for top, right, bottom, left in face_locations]

        return face_locations, face_names

