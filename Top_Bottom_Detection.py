from ultralytics import YOLO

# Load Top&bottom Detection Model
TB_Detection_Model = YOLO('TOP&BOTTOM_Detection.pt')


# Load Image
Fashion_Image = './test_set'

# Save Result(Top Bottom Detection)
Top_Bottom_Detection_Result = 'Top_Bottom_Detection_Result'

# Predict Top&Bottom
result = TB_Detection_Model.predict(source='Top_Bottom_Combination', save=True, project=Top_Bottom_Detection_Result, device='mps')