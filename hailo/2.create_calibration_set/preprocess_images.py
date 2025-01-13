import os
import cv2

# Paths
input_dir = "data"  # Replace with your actual input directory
output_dir = "processed_faces"
os.makedirs(output_dir, exist_ok=True)

# Load Haar Cascade for face detection
frontal_face_cascade_path = '../../haarcascades/haarcascade_frontalface_alt.xml'  # Replace with your actual path
face_cascade = cv2.CascadeClassifier(frontal_face_cascade_path)

# Process each image in the input directory
images_list = [img_name for img_name in os.listdir(input_dir) if os.path.splitext(img_name)[1].lower() == ".jpg"]

for img_name in sorted(images_list):
    # Open and convert the image to grayscale
    img_path = os.path.join(input_dir, img_name)
    frame = cv2.imread(img_path)
    grayscale_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(grayscale_image, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # Process each detected face
    for i, (x, y, w, h) in enumerate(faces):
        face = grayscale_image[y:y + h, x:x + w]  # Crop the face

        # Resize to 224x224 using OpenCV
        face_resized = cv2.resize(face, (224, 224))

        # Save the processed face image as grayscale
        output_face_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_face_{i}.png")
        cv2.imwrite(output_face_path, face_resized)

print(f"Processed face images saved to {output_dir}")
