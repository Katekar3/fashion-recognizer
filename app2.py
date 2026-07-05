import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
        self.flatten = nn.Flatten()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
    

model = CNN()
model.load_state_dict(torch.load('fashion_cnn.pth'))
model.eval()

st.title("FASHION RECOGNISER")

uploaded_file = st.file_uploader("Upload Image", type=['jpeg','png','jpg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", width=150)
    img = img.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img)/255.0
    image_tensor = torch.tensor(data=img_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    if st.button("Predict"):    
        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs, 1)
        st.write(f"Predicted Output: {class_names[predicted.item()]}")