from ultralytics import YOLO


# Load YOLO Model best.pt Path
Load_Model_Path = 'Train_Result/train9/weights/best.pt'

model = YOLO(Load_Model_Path)

# Test Clothing Image Path
Predict_Images_Path = 'test_set'

# Predict Clothing Image Path
Top_Bottom_Combination = 'Top_Bottom_Combination'

Insta_Images_Path = 'Insta_images'

# One Image Path
# One_Image_Path = '/Users/seunghunjang/Desktop/WOOTD_2/test_Combination/test_23.png'

# Save Predict Result Path
Predict_Result_Path = 'Predict_Result'

# Predict Model
result_Combination = model.predict(source=Top_Bottom_Combination, save=True, save_txt=True, project=Predict_Result_Path, device='mps')
# result_Combination = model.predict(source=One_Image_Path, device='mps')
