# Respiratory Disease Classification

This project focuses on classifying chest X-ray images into five categories: **COVID-19**, **Pneumonia**, **Tuberculosis**, **Lung Opacity**, and **Normal**. The classification is performed using a deep learning model based on **ResNet-18**, trained via transfer learning. The project includes both the training notebooks and a web application for real-time predictions.

---

## Project Structure

The project is divided into two main parts:
1. **Notebooks**: Contains Jupyter notebooks for data preprocessing, model training, evaluation, and inference.
2. **App**: Contains the FastAPI-based web application for real-time predictions.

### Notebooks
- **Data Preprocessing**: Loads and preprocesses the dataset for training.
- **Model Training**: Defines and trains the ResNet-18 model using transfer learning.
- **Model Evaluation**: Evaluates the model's performance on the test dataset.
- **Inference**: Demonstrates how to use the trained model for predictions.

### App
- **FastAPI Backend**: Handles image uploads and predictions.
- **Frontend**: A simple HTML interface for uploading images and viewing results.
- **Docker**: The app is containerized using Docker for easy deployment.

---

## Dataset

The dataset used for training and evaluation is **Chest X-Ray Dataset for Respiratory Disease Classification** by Basu et al. (2021). Due to its large size, the raw dataset is not included in this repository. However, you can download it from the following link:

👉 [https://doi.org/10.7910/DVN/WNQ3GI](https://doi.org/10.7910/DVN/WNQ3GI)

### Citation
If you use this dataset, please cite it as follows:

Basu, Arkaprabha; Das, Sourav; Ghosh, Susmita; Mullick, Sankha; Gupta, Avisek; Das, Swagatam, 2021, "Chest X-Ray Dataset for Respiratory Disease Classification", https://doi.org/10.7910/DVN/WNQ3GI, Harvard Dataverse, V5.

---

## Requirements

To run the notebooks or the web app locally, you will need the following dependencies:
- Python 3.8+
- PyTorch
- FastAPI
- Uvicorn
- Pillow
- NumPy

You can install the required packages using the `requirements.txt` file:

pip install -r requirements.txt

---

## Running the Web App Locally

1. Clone the repository:

git clone https://github.com/sgessesse/Respiratory_Disease_Classification.git

cd Respiratory_Disease_Classification

2. Install dependencies:

pip install -r requirements.txt

3. Download the trained model (my_trained_model.pth) and place it in the model folder.

4. Run the FastAPI app:

uvicorn main:app --reload

5. Open your browser and navigate to http://127.0.0.1:8000/ to access the web app.

---

## Deployment

The web app is deployed on AWS Elastic Beanstalk using Docker. The Dockerfile and deployment configuration files are included in the app folder.

---

## Limitations

- The model's accuracy may vary depending on the quality and source of the X-ray images.
- The dataset used for training is limited to specific types of X-ray machines, which may affect generalization to other sources.

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
- **PyTorch**: For the deep learning framework.
- **Kaiming He, et al.:**: for their work on the ResNet architecture.

---

## Contact

For questions or feedback, feel free to reach out:
📧 [sem_werede@yahoo.com]
🌐 (https://github.com/sgessesse)