import os
import cv2
import json
from flask import Flask, request, jsonify, render_template
import base64
from google.cloud import aiplatform
from google.cloud.aiplatform.gapic.schema import predict

app = Flask(__name__, static_folder='static')

@app.route('/')
def root():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    predictions=[]
    img=""
    successful_upload = False
    if request.method == 'POST':
        uploaded_file = request.files.get('picture')
        if uploaded_file:
            img = uploaded_file.filename
            file_path = os.path.join("/home/shirley_acebop/proyecto/app/static/input", uploaded_file.filename)
            uploaded_file.save(file_path)
            # Hacer la predicción con el modelo
            project = "1062486954391"
            endpoint_id = "8941065829353521152"
            location = "us-central1"
            predictions = predict_image_object_detection_sample(
                project, endpoint_id, file_path, location)
            # Leer la salida de la predicción
            output_file_path = os.path.join("/home/shirley_acebop/proyecto/app/static/input", "input.txt")
            with open(output_file_path, "r") as f:
                output = f.read()
            dibujar(predictions, img)
            # Mostrar la salida en la página
            successful_upload = True
    return render_template('upload_photo.html', 
                           successful_upload=successful_upload,
                           predictions=predictions,
                           img=img)

def predict_image_object_detection_sample(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(
        client_options=client_options)
    with open(filename, "rb") as f:
        file_content = f.read()

    # The format of each instance should conform to the deployed model's prediction input schema.
    encoded_content = base64.b64encode(file_content).decode("utf-8")
    instance = predict.instance.ImageObjectDetectionPredictionInstance(
        content=encoded_content,
    ).to_value()
    instances = [instance]
    # See gs://google-cloud-aiplatform/schema/predict/params/image_object_detection_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageObjectDetectionPredictionParams(
        confidence_threshold=0.5, max_predictions=5,
    ).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/image_object_detection_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))

    return dict(prediction)

def dibujar(pred, imgname):
    # Lee la imagen
    img = cv2.imread(os.path.join("/home/shirley_acebop/proyecto/app/static/input",imgname))

    # Dibuja las regiones de interés en la imagen
    prediction = pred

    for bbox in prediction['bboxes']:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1 * img.shape[1]), int(y1 * img.shape[0]), int(x2 * img.shape[1]), int(y2 * img.shape[0])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Guarda la imagen en la carpeta "output"
    cv2.imwrite(os.path.join("/home/shirley_acebop/proyecto/app/static/output",imgname), img)


if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=int(os.environ.get('PORT', 8080)))
