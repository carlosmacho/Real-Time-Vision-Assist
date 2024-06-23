import os
import subprocess
from roboflow import Roboflow
from IPython.display import Image, display
import requests
import torch
from dotenv import load_dotenv

# Take environment variables from .env
load_dotenv()

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
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
            "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
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
        ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY')
        rf = Roboflow(api_key="ROBOFLOW_API_KEY")

        # Access the specific project and version
        project = rf.workspace("my-workspace-zdfxw").project("yolo-real-time-object-detection")
        version = project.version(5)
        
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
        yolo_model_path = os.path.join(HOME, "weights", "yolov8n.pt")
        dataset_location = "C:\\Users\\Carlos\\vsProjects\\AOOP-CP3\\Yolo-Real-Time-Object-Detection-5" # Update this with the actual dataset location
        #dataset_location = os.path.join(HOME, "datasets") # Update this with the actual dataset location

        # Run the YOLO training command
        command = [
            'yolo',
            'task=detect',
            'mode=train',
            'epochs=300',
            'patience=100',
            'batch=8',
            'plots=True',
            f'model={yolo_model_path}',
            f'data={dataset_location}\data.yaml',
            'imgsz=640',
            'device=cuda' # Explicitly specify GPU usage if available
        ]
        subprocess.run(command, check=True)
        print_success("YOLO training completed successfully.")
        return True

    except subprocess.CalledProcessError:
        print_failure("Error during YOLO training.")
        return False
    
def main():
    steps = {
        1: check_gpu,
        2: download_weights,
        3: download_dataset,
        4: train_yolo,
    }

    while True:
        print("\n==========================================")
        print("            YOLO Training Menu           ")
        print("==========================================\n")
        print("Choose a step to run (1-8), or '0' to exit:\n")
        print("1. Check GPU")
        print("2. Download weights")
        print("3. Download dataset")
        print("4. Train YOLO")
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
                print("\nInvalid choice. Please enter a number between 0 and 4.")
        except ValueError:
            print("\nInvalid choice. Please enter a number.")

if __name__ == "__main__":
    main()
