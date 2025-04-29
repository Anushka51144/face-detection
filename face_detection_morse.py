import cv2
import time

# Morse code mapping
morse_code_mapping = {
    1: '.',  # 1 blink detected
    2: '-',  # 2 lips detected
    3: ' ',  # 3 face detected (space)
}

# Morse code to letter mapping
morse_to_text = {
    '.-': 'A', '-...': 'B', '-.-.': 'C', '-..': 'D', '.': 'E',
    '..-.': 'F', '--.': 'G', '....': 'H', '..': 'I', '.---': 'J',
    '-.-': 'K', '.-..': 'L', '--': 'M', '-.': 'N', '---': 'O',
    '.--.': 'P', '--.-': 'Q', '.-.': 'R', '...': 'S', '-': 'T',
    '..-': 'U', '...-': 'V', '.--': 'W', '-..-': 'X', '-.--': 'Y',
    '--..': 'Z', ' ': ' '
}

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture from the webcam
cap = cv2.VideoCapture(0)

# Initialize Morse code string and decoded message
morse_code = ""
decoded_message = ""
last_signal_time = time.time()
timeout = 5  # Timeout in seconds to decode automatically

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Count the number of detected faces
    num_faces = len(faces)

    # Update Morse code based on the number of detected faces
    if num_faces in morse_code_mapping:
        morse_code += morse_code_mapping[num_faces]
        last_signal_time = time.time()  # Reset the timeout timer

    # Automatically decode Morse code after a timeout
    if time.time() - last_signal_time > timeout and morse_code:
        # Split the Morse code into words and decode
        words = morse_code.strip().split(' ')
        for word in words:
            if word in morse_to_text:
                decoded_message += morse_to_text[word]
        morse_code = ""  # Reset Morse code after decoding

    # Decode Morse code into text when the user signals a space (3 blinks)
    if num_faces == 3 and morse_code:
        # Split the Morse code into words and decode
        words = morse_code.strip().split(' ')
        for word in words:
            if word in morse_to_text:
                decoded_message += morse_to_text[word]
        morse_code = ""  # Reset Morse code after decoding

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the Morse code and decoded message on the frame
    cv2.putText(frame, f'Morse Code: {morse_code}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f'Decoded: {decoded_message}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection Morse Code', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()