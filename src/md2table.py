import pandas as pd


def extract_table_from_markdown(markdown_str):
    """
    Extracts a table from a Markdown-style string and returns it as a Pandas DataFrame.

    Parameters:
        markdown_str (str): The input string containing the table in Markdown-like format.

    Returns:
        pd.DataFrame: The extracted table as a DataFrame.
    """
    # Split the input string by lines
    rows = markdown_str.split("\n")

    # List to store the table data
    table_data = []

    # Loop through rows and extract data between the pipes ('|')
    for row in rows:
        if "|" in row:  # Identify rows containing table data
            # Split the row by '|' and strip whitespace from each cell
            cells = [cell.strip() for cell in row.split("|") if cell.strip()]
            table_data.append(cells)

    # The first row is the header, and the subsequent rows contain data
    header = table_data[0]
    data = table_data[2:]  # Skip the separator row (e.g., |---|---|---|)

    # Create and return the DataFrame
    df = pd.DataFrame(data, columns=header)
    # Convert appropriate columns to numeric (float), ignoring the first column (Skill Names)
    for col in df.columns[1:]:  # Skip the first column as it contains skill names
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
