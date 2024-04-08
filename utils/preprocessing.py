import json
import numpy as np
import pandas as pd
import re

from tqdm import tqdm
from yaml import safe_load


def load_entities(filename: str) -> dict:
    """Loads entities data from a JSON file.

    Args:
        filename: Path to the JSON file.

    Returns:
        A dictionary representing the loaded entities data.
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found: {filename}")
        return {}


def load_tokens(filename: str) -> dict:
    """Loads token data from a YAML file.

    Args:
        filename: Path to the YAML file.

    Returns:
        A dictionary representing the loaded token data.
    """
    try:
        with open(filename, "r") as f:
            return safe_load(f)
    except FileNotFoundError as e:
        print(f"Error: File not found: {filename}")
        return {}


def get_token_dict(tokens):
    """Creates a dictionary mapping token start characters to their corresponding entries.

    Args:
        tokens (dict): A dictionary where keys are entries and values are dictionaries containing token information.

    Returns:
        dict: A dictionary mapping token start characters to their entries.
    """

    token_dict = {}
    for entry in tokens.keys():
        token = tokens[entry]["start"]
        token_dict[token] = entry
    return token_dict


def create_empty_entry(tokens: dict) -> dict:
    """
    Create an empty entry dictionary with the given tokens as keys.

    Parameters:
    tokens (dict): A dictionary of tokens.

    Returns:
    dict: An empty entry dictionary with the given tokens as keys.
    """
    return {token: None for token in tokens.keys()}


def split_by_token(token_dict: dict, line: str) -> list[str]:
    """Splits a line based on token patterns.

    Args:
        line: The line to split.

    Returns:
        A list of tokens and corresponding values, with empty strings filtered out.
    """
    # Precompile the regular expression for efficiency
    token_pattern = re.compile(
        "(" + "|".join(re.escape(token) for token in token_dict.keys()) + ")"
    )

    splits = token_pattern.split(line)
    return [part.strip() for part in splits if part]


def split_to_dict(token_dict: dict, split: list[str], dict_split: dict = None) -> dict:
    """Converts a list of tokens and values into a dictionary.

    Args:
        split: A list containing alternating tokens and values.
        dict_split: Optional dictionary to populate (defaults to a new empty dict).

    Returns:
        A dictionary with tokens as keys and corresponding values.

    Raises:
        ValueError: If the length of the split list is odd.
    """
    # Ensure an even number of elements
    if len(split) % 2 != 0:
        raise ValueError("Invalid line format: Expected even number of elements")

    # Initialize the dictionary if not provided
    if dict_split is None:
        dict_split = {}

    # Iterate over pairs of elements
    for i in range(0, len(split), 2):
        token = split[i]
        element = split[i + 1]
        dict_split[token_dict[token]] = element
    return dict_split


def process_entities_data(token_dict: dict, entities: dict) -> pd.DataFrame:
    """
    Process the entities data and convert it into a pandas DataFrame.

    Args:
        tokens (dict): A dictionary containing the tokens and their corresponding values.
        entities (dict): A dictionary containing the entity data.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the processed entity data.

    Raises:
        ValueError: If there is an error processing an entry.

    """
    df_dict = {}
    index = 0

    # Iterate over each key in the entities dictionary
    for key in tqdm(entities.keys()):
        # Iterate over each entry in the entity data
        for entry in entities[key].split("\n"):
            try:
                # Split the line into tokens and values
                split = split_by_token(token_dict, entry)
                # Convert the splited list into a dictionary
                split_dict = split_to_dict(token_dict, split)
                # Store the dictionary in the main data dictionary
                df_dict[index] = split_dict
                index += 1
            except ValueError as e:
                print(f"Error processing entry: {entry}")
                print(e)

    # Create a DataFrame from the dictionary
    df = pd.DataFrame().from_dict(df_dict, orient="index").fillna(value=np.nan)

    # Remove rows with all NaN values
    indices_to_remove = []
    for i in range(len(df)):
        if np.all(df.iloc[i].isna()):
            indices_to_remove.append(i)
    df = df.loc[~df.index.isin(indices_to_remove)].reset_index(
        drop=True
    )  # Reset index after removing rows

    return df
