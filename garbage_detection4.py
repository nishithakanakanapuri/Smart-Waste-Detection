import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util, visualization_utils as vis_util, ops as utils_ops
import smtplib
import io
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart

# Email sending function with image attachment
def send_email(image_with_boxes):
    try:
        # Convert the processed image (with bounding boxes) to a byte stream
        img_byte_arr = io.BytesIO()
        Image.fromarray(image_with_boxes).save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Set up the email as a multipart message
        msg = MIMEMultipart()
        msg['Subject'] = "Garbage Detection Alert"
        msg['From'] = "nishithakanakanapuri@gmail.com"
        msg['To'] = "nishithakanakanapuri@gmail.com"

        # Attach the text part
        body = MIMEText("Trash Detected")
        msg.attach(body)

        # Attach the image part
        image_part = MIMEImage(img_byte_arr.read(), name="detected_image.png")
        msg.attach(image_part)

        # Send the email
        with smtplib.SMTP('smtp.gmail.com', 587) as s:
            s.starttls()
            s.login("nishithakanakanapuri@gmail.com", "bczjstmynzecvugw")  # Use your app password
            s.sendmail(msg['From'], msg['To'], msg.as_string())

        messagebox.showinfo("Notification", "Alert email sent with image!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to send email: {e}")

# Path configurations
PATH_TO_FROZEN_GRAPH = "inference_graph/frozen_inference_graph.pb"
PATH_TO_LABELS = "training/labelmap.pbtxt"

# Load the detection graph
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def process_frame(frame, email_sent):
    output_dict = run_inference_for_single_image(frame, detection_graph)
    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.75
    )
    if not email_sent and len(output_dict['detection_classes']) > 0 and max(output_dict['detection_scores']) > 0.75:
        send_email(frame)  # Send the image with bounding boxes
        email_sent = True
    return frame, email_sent

def upload_picture():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        image = Image.open(file_path)
        image_np = load_image_into_numpy_array(image)
        output_dict = run_inference_for_single_image(image_np, detection_graph)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8
        )
        if len(output_dict['detection_classes']) > 0 and max(output_dict['detection_scores']) > 0.7:
            send_email(image_np)  # Send the processed image
        cv2.imshow("Image Detection", image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if file_path:
        cap = cv2.VideoCapture(file_path)
        email_sent = False  # Prevent multiple emails for the same video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame, email_sent = process_frame(frame, email_sent)
            frame_height, frame_width = frame.shape[:2]
            cv2.namedWindow("Video Detection", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Video Detection", frame_width, frame_height)  # Resize the window to fit the frame
            cv2.imshow("Video Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        cap.release()
        cv2.destroyAllWindows()

def use_webcam():
    cap = cv2.VideoCapture(0)
    email_sent = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame, email_sent = process_frame(frame, email_sent)
        frame_height, frame_width = frame.shape[:2]
        cv2.namedWindow("Webcam Detection", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Webcam Detection", frame_width, frame_height)  # Resize the window to fit the frame
        cv2.imshow("Webcam Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Garbage Detection System")

frame = tk.Frame(root)
frame.pack(pady=20)

upload_image_button = tk.Button(frame, text="Upload Image", command=upload_picture, width=20)
upload_image_button.grid(row=0, column=0, padx=10, pady=10)

upload_video_button = tk.Button(frame, text="Upload Video", command=upload_video, width=20)
upload_video_button.grid(row=0, column=1, padx=10, pady=10)

webcam_button = tk.Button(frame, text="Use Webcam", command=use_webcam, width=20)
webcam_button.grid(row=0, column=2, padx=10, pady=10)

root.mainloop()
