import pandas as pd
from datasets import load_dataset

def main():
    print("Streaming dataset from Hugging Face: adybacki/bu_green_line_ml_ready...")
    
    # Load the dataset using streaming=True so it doesn't download the whole file
    dataset = load_dataset("adybacki/bu_green_line_ml_ready", split="train", streaming=True)

    print("\nFetching the first 10 entries from the stream...")
    # Grab just the first 10 rows from the stream
    first_10 = list(dataset.take(10))
    
    print("\nConverting to Pandas DataFrame...")
    df = pd.DataFrame(first_10)

    print("\nFirst 10 entries of the dataset:")
    # Set pandas display options to make sure we can see all the columns when printing
    pd.set_option('display.max_columns', None)
    print(df)

if __name__ == "__main__":
    main()
