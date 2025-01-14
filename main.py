print("Starting FastAPI application...")

import PIL
import pandas as pd
import torch
import sys
import os
import cv2
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, Response
from PIL import Image
import io
import torchvision.transforms as transforms
from fastapi.templating import Jinja2Templates
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from torch.profiler import profile, record_function, ProfilerActivity
from Utils.utils import build_annotation_dataframe, check_annot_dataframe, transform_bilinear, transform_bilinear_validate, infer, infer_single_image, calculate_model_performance, generate_fn_cost_matrix, generate_fp_cost_matrix, get_current_timestamp
from models import get_model
# Jinja2 templates setup
templates = Jinja2Templates(directory="templates")

print("Imports done")

sys.path.append('/zhome/ac/d/174101/thesis/src')
import torchvision.datasets as datasets


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device set")




app = FastAPI()

# Ensure the 'static' directory exists for serving static files
if not os.path.exists('static'):
    os.makedirs('static')

# Mount the static directory to serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

base_path = 'models/DataSetLast2Days/'
model_name = "DataSetLast2Days_vit_b_16_model.pth" #specify a model
model_path = Path(base_path) / f"{model_name}" 

train_data = "models/DataSetLast2Days/train.csv"
test_data = 'models/DataSetLast2Days/test.csv'
val_data = "models/DataSetLast2Days/val.csv"
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)
val_df = pd.read_csv(val_data)
all_df = pd.concat([train_df, test_df, val_df], ignore_index=True)  




all_class_names = list(all_df['class_name'].unique())
all_num_classes = len(all_class_names)

print("Datasets loaded")
all_model = get_model("vit_b_16", all_num_classes)
all_model = all_model.to(device)
all_model.load_state_dict(torch.load(model_path, map_location=device))

base_path = 'models/DataSetCutLast2Days/'
model_name = "DataSetCutLast2Days_vit_b_16_model.pth" #specify a model
model_path = Path(base_path) / f"{model_name}" 

train_data = "models/DataSetCutLast2Days/train.csv"
test_data = 'models/DataSetCutLast2Days/test.csv'
val_data = "models/DataSetCutLast2Days/val.csv"
train_df = pd.read_csv(train_data)
test_df = pd.read_csv(test_data)
val_df = pd.read_csv(val_data)
all_df = pd.concat([train_df, test_df, val_df], ignore_index=True)  




six_class_names = list(all_df['class_name'].unique())
six_num_classes = len(six_class_names)

print("Datasets loaded")
six_model = get_model("vit_b_16", six_num_classes)
six_model = six_model.to(device)
six_model.load_state_dict(torch.load(model_path, map_location=device))


# Prediction endpoint
@app.post("/predict/", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        all_model.eval()
        image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224),
                          interpolation=PIL.Image.BILINEAR),


        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((500,750))
        image_path = f"static/{file.filename}"
        image.save(image_path)
        
        image = cv2.imread(image_path)  # Read image using cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = image_transform(image)
        print("test1")
        # Display the transformed image (optional)
        image_transformed_sq = torch.unsqueeze(image_transformed, dim=0)
        print("test1323")
        with torch.no_grad():
            image_transformed_sq = image_transformed_sq.to(device)
            outputs = all_model(image_transformed_sq)
            print("test22")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # probabilities = torch.sigmoid(outputs)
            print("test223")   
            top_probabilities, top_indices = torch.topk(probabilities, 5)
        print("test2")
        
        # Inference
        # Convert to numpy and make the output more readable
        top_probabilities = top_probabilities.cpu().numpy().flatten() * 100  # Convert to percentage
        top_indices = top_indices.cpu().numpy().flatten()
        print(top_probabilities)
        predictions = []
        print("test3")
        
        for i in range(len(top_probabilities)):
            predictions.append({
                "class_name": all_class_names[top_indices[i]],
                "probability": top_probabilities[i].item()
        })
        # Print the results
   
        # Render predictions.html template with predictions
        return templates.TemplateResponse("predictions.html", {"request": request, "predictions": predictions, "image_path": f"/static/{file.filename}"})
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return {"error": "Internal Server Error"}

@app.post("/predictsixplate/", response_class=HTMLResponse)
async def predictsixplate(request: Request, file: UploadFile = File(...)):
    try:
        six_model.eval()
        image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224),
                          interpolation=PIL.Image.BILINEAR),


        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((500,750))
        image_path = f"static/{file.filename}"
        image.save(image_path)
        
        image = cv2.imread(image_path)  # Read image using cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = image_transform(image)
        print("test1")
        # Display the transformed image (optional)
        image_transformed_sq = torch.unsqueeze(image_transformed, dim=0)
        print("test1323")
        with torch.no_grad():
            image_transformed_sq = image_transformed_sq.to(device)
            outputs = six_model(image_transformed_sq)
            print("test22")
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            # probabilities = torch.sigmoid(outputs)
            print("test223")   
            top_probabilities, top_indices = torch.topk(probabilities, 5)
        print("test2")
        
        # Inference
        # Convert to numpy and make the output more readable
        top_probabilities = top_probabilities.cpu().numpy().flatten() * 100  # Convert to percentage
        top_indices = top_indices.cpu().numpy().flatten()
        print(top_probabilities)
        predictions = []
        print("test3")
        
        for i in range(len(top_probabilities)):
            predictions.append({
                "class_name": six_class_names[top_indices[i]],
                "probability": top_probabilities[i].item()
        })
        # Print the results
   
        # Render predictions.html template with predictions
        return templates.TemplateResponse("predictions.html", {"request": request, "predictions": predictions, "image_path": f"/static/{file.filename}"})
    
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return {"error": "Internal Server Error"}


print("Prediction endpoint defined")

# Serve the HTML form for uploading images
@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    content = """
    <html>
    <head>
        <title>Upload Image</title>
    </head>
    <body>
        <h2>Normal six plate</h2>
        <form action="/predict/" enctype="multipart/form-data" method="post">
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        <h2>Six plate divided, only one plate</h2>
         <form action="/predictsixplate/" enctype="multipart/form-data" method="post">
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
    </body>
    </html>
    """
    return HTMLResponse(content=content)

print("Main endpoint defined")

if __name__ == "__main__":
    print("Running app with uvicorn")
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
