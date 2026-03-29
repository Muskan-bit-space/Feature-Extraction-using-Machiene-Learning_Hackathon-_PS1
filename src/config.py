# config.py

CLASS_MAP = {
    "Bridge":            0,           
    "Built_Up_Area_type":1,
    "Railway":           2,
    "Road":              3,
    "Roal_Centre_Line":  4,
    "Utlity":            5,
    "Utility_Poly_":     6,
    "Water_Body":        7,
    "Water_Body_Line":   8,
    "Waterbody_Point":   9

}

PATCH_SIZE = 256
STRIDE = 256

TARGET_CRS = "EPSG:4326"
#3857

DATA_DIR = "D:\/ML_Hackathon\/data\/PB_training_dataSet_shp_file\/"
#"D:\ML_Hackathon\data\/PB_training_dataSet_shp_file\/NADALA_28996_ORTHO.tif"
OUTPUT_DIR = "C:\/My_D_Drive\/Customers\/Source_Code\/DataScience_Machine_Learning\/feature_extraction_from_tif\/ortho\/output\/"

PREDICT_DIR= "D:\/ML_Hackathon\/test_data\/PB_live_demo\/live_demo_3\/"

EPOCHS = 45
LR = 1e-3
BATCH_SIZE = 4

TEST_PATCH_SIZE = 512
TEST_STRIDE = 512