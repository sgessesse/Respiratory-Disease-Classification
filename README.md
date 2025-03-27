# Respiratory Disease Classification

This project focuses on classifying chest X-ray images into five categories: **COVID-19**, **Pneumonia**, **Tuberculosis**, **Lung Opacity**, and **Normal**. The classification is performed using a deep learning model based on **ConvNeXt-Tiny**, trained via transfer learning. The project includes both the training scripts and a web application for real-time predictions.

---

## Live Demo

A live demo of the deployed application is available at:

👉 [https://d3rh0b7i52zqpl.cloudfront.net/](https://d3rh0b7i52zqpl.cloudfront.net/)

*(Note: This demo runs the inference application.)*

---

## Project Structure

The project is divided into two main parts:
1. **Training**: Contains Python scripts for data preprocessing, model definition, training, evaluation, and prediction logic.
2. **App**: Contains the FastAPI-based web application for real-time predictions, including the trained model.

### Training (`training/`)
- `preprocessing.py`: Loads and preprocesses the dataset, including augmentations (CLAHE, ColorJitter) and normalization.
- `model.py`: Defines the ConvNeXt-Tiny model architecture.
- `train.py`: Handles the model training loop with AdamW, CosineAnnealingLR, gradient clipping, and label smoothing.
- `evaluate.py`: Evaluates the model's performance (Accuracy, Precision, Recall, F1, Confusion Matrix) and saves results.
- `run_training_pipeline.py`: Orchestrates the preprocessing, training, evaluation, and model saving pipeline. (Formerly `main.py`)
- `requirements.txt`: Dependencies required for running the training pipeline.
- `evaluation_results/`: Directory where evaluation metrics (CSV) and plots (PNG) are saved.

### App (`app/`)

### App
- `main.py`: FastAPI backend handling image uploads and predictions.
- `predict.py`: Contains the `Predictor` class for preprocessing input images and running inference.
- `static/index.html`: HTML frontend for uploading images and viewing results.
- `model/my_trained_model.pth`: The trained ConvNeXt-Tiny model weights. *(Note: This file is excluded by `.gitignore` due to its size and must be generated by running the training pipeline).*
- `requirements.txt`: Dependencies required *only* for running the prediction app.
- `Dockerfile`: Configuration for building the Docker image for deployment.

---

## Dataset

The dataset used for training and evaluation is **Chest X-Ray Dataset for Respiratory Disease Classification** by Basu et al. (2021). Due to its large size, the raw dataset is not included in this repository. However, you can download it from the following link:

👉 [https://doi.org/10.7910/DVN/WNQ3GI](https://doi.org/10.7910/DVN/WNQ3GI)

### Citation
If you use this dataset, please cite it as follows:

Basu, Arkaprabha; Das, Sourav; Ghosh, Susmita; Mullick, Sankha; Gupta, Avisek; Das, Swagatam, 2021, "Chest X-Ray Dataset for Respiratory Disease Classification", https://doi.org/10.7910/DVN/WNQ3GI, Harvard Dataverse, V5.

---

## Requirements

To run the project components, you will need Python 3.9+ and several libraries.

**For the Prediction App (`app/`):**
The necessary packages are listed in `app/requirements.txt`. Install them using:
```bash
pip install -r app/requirements.txt
```
This includes: `fastapi`, `uvicorn`, `torch`, `torchvision`, `Pillow`, `python-multipart`, `numpy`.

**For the Training Scripts (`training/`):**
All necessary packages (including app dependencies) are listed in `training/requirements.txt`. Install them using:
```bash
pip install -r training/requirements.txt
```
This includes libraries like `opencv-python`, `scikit-learn`, `pandas`, `matplotlib`, `seaborn` in addition to the core app dependencies.

---

## Running the Web App Locally

1. Clone the repository:

git clone https://github.com/sgessesse/Respiratory_Disease_Classification.git

cd Respiratory_Disease_Classification

2. Install app dependencies:
```bash
pip install -r app/requirements.txt
```
3. **Generate the Model:** Run the training pipeline to generate the `my_trained_model.pth` file. This requires installing training dependencies first:
   ```bash
   # Install training dependencies
   pip install -r training/requirements.txt
   # Run the training pipeline (this will take time and requires the dataset)
   python training/run_training_pipeline.py
   ```
   This will save the trained model to `app/model/my_trained_model.pth`.

4. **Run the App:** Run the FastAPI app from the project root directory:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
   *(Note: The Dockerfile uses port 80, but running locally often uses 8000)*

5. Open your browser and navigate to `http://127.0.0.1:8000/` to access the web app.

---

## Deployment

The web app is deployed on AWS Elastic Beanstalk using Docker. The Dockerfile and deployment configuration files are included in the app folder.

---

## Limitations

- The model's accuracy may vary depending on the quality and source of the X-ray images.
- The dataset used for training is limited to specific types of X-ray machines, which may affect generalization to other sources. Performance on images significantly different from the training data (e.g., low-quality screenshots) is not guaranteed.
- **Disclaimer:** The predictions provided by this application are experimental and for informational purposes only. They are **not** a substitute for professional medical diagnosis. Always consult a qualified radiologist or healthcare provider.

---

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

---

## License

This project is licensed under the MIT License. 

---

## Acknowledgments

- **Dataset**: Basu et al. (2021) for providing the Chest X-Ray Dataset.
- **FastAPI**:  For the web framework.
- **PyTorch & Torchvision**: For the deep learning framework and models.
- **ConvNeXt Authors**: For the ConvNeXt architecture.

---

## Contact

For questions or feedback, feel free to reach out:
📧 [semir.w.gessesse@gmail.com]
🌐 (https://github.com/sgessesse)
