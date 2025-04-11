from utils import download_file, extract_file
from config import DATASET_URL

def main():
    output_path = download_file(DATASET_URL)
    extract_file(output_path)

if __name__ == "__main__":
    main()