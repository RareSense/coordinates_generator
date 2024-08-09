import json
import gradio as gr
from PIL import Image
import numpy as np
from scipy import ndimage

# Load each line of the file as a separate JSON object
data = []
with open('master.json', 'r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# Global variable to keep track of the current index
current_index = 0

# Function to load progress
def load_progress():
    global current_index
    try:
        with open('progress.json', 'r') as progress_file:
            progress_data = json.load(progress_file)
            current_index = progress_data.get('current_index', 0)
    except FileNotFoundError:
        current_index = 0  # Start from the beginning if no progress is saved

# Function to save progress
def save_progress(index):
    progress_data = {
        'current_index': index
    }
    with open('progress.json', 'w') as progress_file:
        json.dump(progress_data, progress_file)

# Function to load images
def load_image(index):
    if index < len(data):
        image_path = data[index]['target']
        ai_name= data[index]["ai_name"]
        return Image.open(image_path).convert("RGB"), ai_name
    return None, None

def update_image():
    global current_index
    if current_index >= len(data):
        return None, "Finished! All images processed."

    image, ai_name = load_image(current_index)
    progress_text = f"Current Index: {current_index+1}/{len(data)}"
    
    return image, progress_text, ai_name

def capture_coordinates(input_image):
    if input_image is not None:
        scribble = np.array(input_image["mask"])
        scribble = scribble.transpose(2, 1, 0)[0]
        labeled_array, num_features = ndimage.label(scribble >= 255)
        centers = ndimage.center_of_mass(scribble, labeled_array, range(1, num_features + 1))
        centers = np.array(centers)
        print(f"Coordinates: {centers}")
        return centers
    else:
        print("No coordinates found.")
        return np.array([])

def save_results(coordinates):
    global current_index
    if current_index < len(data):  # Save only if the index is within range
        result_data = {
            "target": data[current_index]['target'],
            "coordinates": coordinates.tolist()
        }
        with open('result.json', 'a') as result_file:
            json.dump(result_data, result_file)
            result_file.write('\n')  # Add a newline for each JSON object

def process_and_update_image(input_image):
    global current_index
    # Capture the coordinates for the current image
    if input_image is not None:
        coordinates = capture_coordinates(input_image)
        # Save the results to result.json
        save_results(coordinates)
        # Increment the index only after processing the current image
        current_index += 1
        save_progress(current_index)
    
    # Load the next image and update the display and progress
    return update_image()

# Load progress when the application starts
load_progress()

with gr.Blocks() as block:
    # No image loaded initially
    image = None
    progress = "Press 'Next' to start processing."

    with gr.Row():
        image_display = gr.Image(interactive=True, tool="sketch", brush_radius=20, width=600, height=600, value=image)
        
    with gr.Row():
        name = gr.Textbox(interactive=False, label="ai_name")
        progress_text = gr.Textbox(label="Progress",value=progress, interactive=False)
        next_button = gr.Button(value="Next")

    next_button.click(
        fn=process_and_update_image,
        inputs=image_display,
        outputs=[image_display, progress_text, name]
    )

block.launch(inbrowser=True)
