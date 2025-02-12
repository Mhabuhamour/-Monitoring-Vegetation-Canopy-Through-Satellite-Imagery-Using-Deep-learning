import streamlit as st
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from sentinelhub import (
    SHConfig,
    DataCollection,
    SentinelHubRequest,
    BBox,
    bbox_to_dimensions,
    CRS,
    MimeType,
)
import torch
from torchvision import transforms
from PIL import Image, ImageEnhance
import numpy as np

# Sentinel Hub configuration
config = SHConfig()
config.sh_client_id = "sh-c67cfdfd-3884-42fa-b1ae-a4b9078f4a54"
config.sh_client_secret = "mZea1VDK53QiiJEnIyeRrEwAyVEroPsP"
config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
config.sh_base_url = "https://sh.dataspace.copernicus.eu"
config.save("cdse")

# Define the wildfire detection model architecture
class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = torch.nn.Linear(64 * 28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 2)  # Binary classification
        self.dropout = torch.nn.Dropout(0.5)  # Dropout layer to prevent overfitting

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 28 * 28)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout before the final layer
        x = self.fc2(x)
        return x

# Load the wildfire detection model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wildfire_model = SimpleCNN().to(device)
wildfire_model.load_state_dict(torch.load(r"C:\Users\user\Desktop\Forest Monitoring\Wildfires_model.pth", map_location=device))
wildfire_model.eval()

# Define the deforestation detection model architecture
class CNNModel(torch.nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.fc1 = torch.nn.Linear(128 * 28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)  # Flatten the output from convolutional layers
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)  # Apply sigmoid for binary classification
        return x

# Load the deforestation detection model
deforestation_model = CNNModel().to(device)
deforestation_model.load_state_dict(torch.load(r"C:\Users\user\Desktop\Forest Monitoring\Deforestation_model.pth", map_location=device))
deforestation_model.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to divide large bounding box into smaller boxes
def divide_bbox(min_long, min_lat, max_long, max_lat, box_size=0.05):
    boxes = []
    long_range = np.arange(min_long, max_long, box_size)
    lat_range = np.arange(min_lat, max_lat, box_size)
    for lon_start in long_range:
        for lat_start in lat_range:
            lon_end = min(lon_start + box_size, max_long)
            lat_end = min(lat_start + box_size, max_lat)
            boxes.append([lon_start, lat_start, lon_end, lat_end])
    return boxes
def get_affected_area_image(coordinates, date, incident_type):
    """
    Retrieve full area image using appropriate evaluation script based on incident type
    """
    min_long, min_lat, max_long, max_lat = coordinates
    aoi_bbox = BBox(bbox=[min_long, min_lat, max_long, max_lat], crs=CRS.WGS84)
    aoi_size = bbox_to_dimensions(aoi_bbox, resolution=10)
    
    # Adjust dimensions if they exceed maximum
    max_dimension = 2500
    if aoi_size[0] > max_dimension:
        scaling_factor = max_dimension / aoi_size[0]
        new_height = int(aoi_size[1] * scaling_factor)
        aoi_size = (max_dimension, new_height)
    if aoi_size[1] > max_dimension:
        scaling_factor = max_dimension / aoi_size[1]
        new_width = int(aoi_size[0] * scaling_factor)
        aoi_size = (new_width, max_dimension)
    
    if incident_type == "Wildfire":
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: ["B02", "B03", "B04", "B08", "B11", "B12", "dataMask"],
                output: { bands: 4 }
            };
        }

        function evaluatePixel(samples) {
            var NDWI=index(samples.B03, samples.B08); 
            var NDVI=index(samples.B08, samples.B04);
            var INDEX= ((samples.B11 - samples.B12) / (samples.B11 + samples.B12))+(samples.B08);

            if((INDEX>0.1)||(samples.B02>0.1)||(samples.B11<0.1)||(NDVI>0.3)||(NDWI > 0.1)){
                return[2.5*samples.B04, 2.5*samples.B03, 2.5*samples.B02, samples.dataMask]
            }
            else {
                return [1, 0, 0, samples.dataMask]
            }
        }
        """
    else:  # Deforestation
        evalscript = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B04", "B08", "B11", "B12"]
                }],
                output: { bands: 3 }
            };
        }

        function evaluatePixel(s) {
            let val = 2.5 * ((s.B11 + s.B04)-(s.B08 + s.B02))/((s.B11 + s.B04)+(s.B08 + s.B02));
            return [2.5* val, s.B08, s.B11];
        }
        """
    
    request = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A.define_from(
                    name="s2l2a", 
                    service_url="https://sh.dataspace.copernicus.eu"
                ),
                time_interval=(date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")),
                other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}
            ),
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=aoi_bbox,
        size=aoi_size,
        config=config,
    )
    
    return request.get_data()[0]
def calculate_area_in_sqkm(coords):
    """
    Calculate the area in square kilometers for a given bounding box.
    
    Args:
        coords (list): Coordinates of the bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat].
        
    Returns:
        float: Area in square kilometers.
    """
    min_lon, min_lat, max_lon, max_lat = coords
    
    # Convert the latitude and longitude differences into kilometers
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    
    # Approximate km per degree for latitude and longitude
    km_per_degree_lat = 111  # 1 degree latitude = 111 km
    km_per_degree_lon = 111 * np.cos(np.radians((min_lat + max_lat) / 2))  # Adjust for longitude based on latitude
    
    # Calculate the area in square kilometers
    area_km2 = lat_diff * km_per_degree_lat * lon_diff * km_per_degree_lon
    return area_km2

def calculate_affected_percentage(image, incident_type, coords):
    """
    Calculate the percentage of the affected area and its size in square kilometers.

    Args:
        image (numpy.ndarray): The image of the affected area.
        incident_type (str): The type of incident ("Wildfire" or "Deforestation").
        coords (list): Coordinates of the bounding box in WGS84 [min_lon, min_lat, max_lon, max_lat].

    Returns:
        tuple: Affected area percentage and affected area in square kilometers.
    """
    # Convert the image to HSV color space for better color segmentation
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define a range for "red" regions (adjust thresholds as needed for the type of incident)
    lower_red1 = np.array([0, 50, 50])    # Lower bound for red hue
    upper_red1 = np.array([10, 255, 255]) # Upper bound for red hue
    lower_red2 = np.array([170, 50, 50])  # Handle wrap-around hue values
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red regions
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # Calculate the percentage of red pixels
    total_pixels = image.shape[0] * image.shape[1]
    red_pixels = np.sum(red_mask > 0)
    red_percentage = (red_pixels / total_pixels) * 100

    # Calculate the total area in square kilometers using the bounding box
    total_area_km2 = calculate_area_in_sqkm(coords)
    
    # Calculate the affected area in square kilometers
    affected_area_km2 = (red_percentage / 100) * total_area_km2
    
    return red_percentage, affected_area_km2

# Streamlit interface
st.title("Forest Vegetation Monitoring")

# Input fields without default values
coordinates_input = st.text_input("Enter Coordinates (min_long, min_lat, max_long, max_lat)")
if coordinates_input:
    coordinates = [float(x) for x in coordinates_input.split(",")]

incident_type = st.selectbox("Select Incident Type", ["Select an Incident Type", "Wildfire", "Deforestation"])

# Date input with no default value (empty field)
date = st.date_input("Select Date", value=None)

if st.button("Retrieve and Classify Image"):
    if coordinates_input and incident_type != "Select an Incident Type" and date:
        min_long, min_lat, max_long, max_lat = coordinates
        small_bboxes = divide_bbox(min_long, min_lat, max_long, max_lat, box_size=0.3)
        time_interval = date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d")

        evalscript_true_color = """
            //VERSION=3
            function setup() {
                return {
                    input: [{ bands: ["B02", "B03", "B04"] }],
                    output: { bands: 3 }
                };
            }
            function evaluatePixel(sample) {
                return [sample.B04, sample.B03, sample.B02];
            }
        """

        incident_detected = False
        for bbox in small_bboxes:
            aoi_bbox = BBox(bbox=bbox, crs=CRS.WGS84)
            aoi_size = bbox_to_dimensions(aoi_bbox, resolution=70)
            max_dimension = 2500
            if aoi_size[0] > max_dimension:
                scaling_factor = max_dimension / aoi_size[0]
                new_height = int(aoi_size[1] * scaling_factor)
                aoi_size = (max_dimension, new_height)
            if aoi_size[1] > max_dimension:
                scaling_factor = max_dimension / aoi_size[1]
                new_width = int(aoi_size[0] * scaling_factor)
                aoi_size = (new_width, max_dimension)
            request_true_color = SentinelHubRequest(
                evalscript=evalscript_true_color,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=DataCollection.SENTINEL2_L2A.define_from(
                            name="s2l2a", service_url="https://sh.dataspace.copernicus.eu"
                        ),
                        time_interval=time_interval,
                        other_args={"dataFilter": {"mosaickingOrder": "leastCC"}}),
                ],
                responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
                bbox=aoi_bbox,
                size=aoi_size,
                config=config,
            )
            true_color_imgs = request_true_color.get_data()
            image = true_color_imgs[0]
            pil_image = Image.fromarray(image)
            enhancer = ImageEnhance.Brightness(pil_image)
            bright_image = enhancer.enhance(1.5)
            input_tensor = transform(bright_image).unsqueeze(0).to(device)
            with torch.no_grad():
                if incident_type == "Wildfire":
                    outputs = wildfire_model(input_tensor)
                    _, predicted = torch.max(outputs, 1)
                    classification = "Wildfire" if predicted.item() == 1 else "No Wildfire"
                elif incident_type == "Deforestation":
                    outputs = deforestation_model(input_tensor)
                    predicted = (outputs > 0.5).float()
                    classification = "Deforestation" if predicted.item() == 1 else "No Deforestation"
            if classification in ["Wildfire", "Deforestation"]:
                incident_detected = True
                st.subheader(f"Retrieved Image for Box: {bbox} - Classification: {classification}")
                st.image(np.array(bright_image), caption=f"Coordinates: {bbox}", use_column_width=True)
                st.write(f"{classification} detected in this area.")
        if not incident_detected:
            st.write(f"No {incident_type.lower()} detected.")
    else:
        st.write("Please provide valid coordinates, select an incident type, and choose a date.")
if st.button("Measure Affected Area"):
    with st.spinner("Retrieving and analyzing affected area..."):
        # Get the full area image using appropriate evaluation script
        full_image = get_affected_area_image(coordinates, date, incident_type)
        # Save the image locally
        output_path = os.path.join("output", f"{incident_type.lower()}_affected_area.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(full_image, cv2.COLOR_RGB2BGR))
        # Pass the saved image to calculate_affected_percentage
        saved_image = cv2.imread(output_path)
        #affected_percentage = calculate_affected_percentage(saved_image, incident_type,coordinates)
        # Unpack the tuple into individual variables
        affected_percentage, affected_area_km2 = calculate_affected_percentage(saved_image, incident_type, coordinates)
        # Display the image with percentage
        st.subheader(f"Affected Area Analysis")
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(full_image)
        #ax.set_title(f'Affected Area: {affected_percentage:.1f}%')
        ax.set_title(f'Affected Area: {affected_percentage:.1f}% ({affected_area_km2:.2f} km²)')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()
        # Display additional information
        st.write(f"Total affected area percentage: {affected_percentage:.2f}%")
        if incident_type == "Wildfire":
            st.write("Analysis based on burned area detection")
        else:
            st.write("Analysis based on barren soil detection")
        

