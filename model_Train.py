from ultralytics import YOLO

# YOLO Model Load
model = YOLO('yolov8n-cls.pt')


# Dataset Path
Dataset_Path = 'Dataset'
Insta_Dataset_Path = 'Insta_dataset'


# Save Tain Model Path
Train_Result = 'Train_Result'

# Save Model Evaluation Path
# Evaluation_Model_Path = 'Train_Result/Model_Evaluations'

# Model Train
result = model.train(data=Insta_Dataset_Path, epochs=40, cache=True, project=Train_Result, device='mps')
