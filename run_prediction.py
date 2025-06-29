from detect_mask import predict_mask

# Run the function on your test image
predict_mask(
    image_path="notebooks/test_face.png",  # or test_face_2.jpg
    model_path="model/face_mask_detector.model"
)
