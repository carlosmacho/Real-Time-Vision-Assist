import os
import subprocess
import cv2
import supervision as sv
from ultralytics import YOLOv10
from roboflow import Roboflow
from IPython.display import Image, display
import requests
import torch


# Get the current working directory
HOME = os.getcwd()

# ANSI color escape codes
class colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'

def print_success(message):
    print(f"{colors.GREEN}[SUCCESS]{colors.END} {message}")

def print_failure(message):
    print(f"{colors.RED}[FAILURE]{colors.END} {message}")

def check_gpu():
    try:
        # Run the nvidia-smi command
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Check if there was an error
        if result.returncode != 0:
            print_failure(f"Error running nvidia-smi: {result.stderr}")
            return False
        else:
            gpu_info = result.stdout
            print(gpu_info)  # Print GPU information
            print_success("GPU information retrieved successfully.")
            return True
    
    except FileNotFoundError:
        print_failure("nvidia-smi command not found. Ensure that the NVIDIA drivers are installed and the PATH is set correctly.")
        return False

def install_packages():
    try:
        # Install the supervision package
        subprocess.run(['pip', 'install', '-q', 'supervision'], check=True)

        # Install the yolov10 package from the GitHub repository
        subprocess.run(['pip', 'install', '-q', 'git+https://github.com/THU-MIG/yolov10.git'], check=True)

        # Install the roboflow package
        subprocess.run(['pip', 'install', 'roboflow'], check=True)

        print_success("Packages installed successfully.")
        return True

    except subprocess.CalledProcessError:
        print_failure("Error installing packages.")
        return False

def download_weights():
    try:
        # Get the current working directory
        HOME = os.getcwd()

        # Define the path to the weights directory within the CP3 folder
        weights_dir = os.path.join(HOME, "weights")

        # Create the weights directory if it doesn't exist
        os.makedirs(weights_dir, exist_ok=True)

        # URLs of the weights files
        urls = [
            "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10n.pt",
            "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10s.pt",
            "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10m.pt",
            "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10b.pt",
            "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10x.pt",
            "https://github.com/jameslahm/yolov10/releases/download/v1.0/yolov10l.pt",
        ]

        # Download the files
        for url in urls:
            file_name = os.path.join(weights_dir, os.path.basename(url))
            with open(file_name, 'wb') as f:
                response = requests.get(url)
                f.write(response.content)

        print_success("Weights downloaded successfully.")
        return True

    except Exception as e:
        print_failure(f"Error downloading weights: {e}")
        return False

def download_dataset():
    try:
        # Initialize Roboflow with the API key
        rf = Roboflow(api_key="JaM7HVDRmrbbCNrYVam0")
        
        # Access the specific project and version
        project = rf.workspace("fyp-sibxz").project("yolov8_door")
        version = project.version(1)
        
        # Download the dataset
        dataset = version.download("yolov8")
        print_success("Dataset downloaded successfully.")
        print(f"Dataset downloaded to: {dataset.location}")
        return True

    except Exception as e:
        print_failure(f"Error downloading dataset: {e}")
        return False

def train_yolo():
    try:
        # Check if GPU is available and print the device information
        device = torch.device("cuda")
        print(f"Using device: {device}")

        # Get the current working directory
        HOME = os.getcwd()

        # Change to the HOME directory
        os.chdir(HOME)
        
        # Define the path to the YOLO model and dataset location
        yolo_model_path = os.path.join(HOME, "weights", "yolov10n.pt")
        dataset_location = "C:\\Users\\Carlos\\CP3\\YOLOv8_Door-1" # Update this with the actual dataset location

        # Run the YOLO training command
        command = [
            'yolo',
            'task=detect',
            'mode=train',
            'epochs=25',
            'batch=8',
            'plots=True',
            f'model={yolo_model_path}',
            f'data={dataset_location}/data.yaml',
            'device=cuda' # Explicitly specify GPU usage if available
        ]
        subprocess.run(command, check=True)
        print_success("YOLO training completed successfully.")
        return True

    except subprocess.CalledProcessError:
        print_failure("Error during YOLO training.")
        return False

def list_directory():
    try:
        # Get the current working directory
        HOME = os.getcwd()

        # Define the path to the directory you want to list
        directory_path = os.path.join(HOME, "runs", "detect", "train3")

        # List the contents of the directory
        if os.path.exists(directory_path):
            for item in os.listdir(directory_path):
                print(item)
            print_success("Directory listed successfully.")
            return True
        else:
            print_failure(f"The directory {directory_path} does not exist.")
            return False

    except Exception as e:
        print_failure(f"Error listing directory: {e}")
        return False

def display_confusion_matrix():
    try:
        # Get the current working directory
        HOME = os.getcwd()

        # Change to the HOME directory
        os.chdir(HOME)
        
        # Define the path to the confusion matrix image
        image_path = os.path.join(HOME, "runs", "detect", "train3", "confusion_matrix.png")
        
        # Display the image
        display(Image(filename=image_path, width=600))
        print_success("Confusion matrix displayed successfully.")
        return True

    except Exception as e:
        print_failure(f"Error displaying confusion matrix: {e}")
        return False

def display_results_image():
    try:
        # Get the current working directory
        HOME = os.getcwd()

        # Change to the HOME directory
        os.chdir(HOME)
        
        # Define the path to the results image
        image_path = os.path.join(HOME, "runs", "detect","train3", "results.png")
        
        # Display the image
        display(Image(filename=image_path, width=600))
        print_success("Results image displayed successfully.")
        return True
   
    except Exception as e:
        print_failure(f"Error displaying results image: {e}")
        return False

def main():
    steps = {
        1: check_gpu,
        2: install_packages,
        3: download_weights,
        4: download_dataset,
        5: train_yolo,
        6: list_directory,
        7: display_confusion_matrix,
        8: display_results_image
    }

    while True:
        print("\n==========================================")
        print("            YOLO Training Menu           ")
        print("==========================================\n")
        print("Choose a step to run (1-8), or '0' to exit:\n")
        print("1. Check GPU")
        print("2. Install packages")
        print("3. Download weights")
        print("4. Download dataset")
        print("5. Train YOLO")
        print("6. List directory")
        print("7. Display confusion matrix")
        print("8. Display results image")
        print("0. Exit")
        print("\n==========================================\n")


        choice = input("Enter your choice: ")

        if choice == '0':
            print("\nExiting the program...")
            break

        try:
            step = int(choice)
            if step in steps:
                print(f"\nRunning step {step}...")
                result = steps[step]()
                if result:
                    print_success(f"Step {step} completed successfully.")
                else:
                    print_failure(f"Step {step} failed.")
            else:
                print("\nInvalid choice. Please enter a number between 0 and 8.")
        except ValueError:
            print("\nInvalid choice. Please enter a number.")

if __name__ == "__main__":
    main()
