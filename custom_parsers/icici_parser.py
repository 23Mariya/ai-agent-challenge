import pandas as pd
import camelot

def parse(pdf_path: str) -> pd.DataFrame:
    """
    Parses a bank statement PDF from 'icici' and returns a pandas DataFrame.

    Args:
        pdf_path: The file path to the ICICI bank statement PDF.

    Returns:
        A pandas DataFrame with the transaction data, conforming to the specified schema.
        Returns an empty DataFrame if no valid transactions can be parsed.
    """
    # 1. Use the `camelot-py` library with `flavor='stream'`.
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
    except Exception:
        # Return an empty DataFrame if Camelot fails to read the PDF
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])

    if not tables:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])

    # 2. Combine tables from all pages of the PDF.
    df = pd.concat([table.df for table in tables], ignore_index=True)

    # Heuristic: Assume the transaction data uses the first 5 columns.
    # If the detected table has fewer than 5 columns, it's not the right one.
    if df.shape[1] < 5:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])

    df = df.iloc[:, :5]

    # 3. The final DataFrame MUST have these columns in this exact order.
    df.columns = ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']

    # 5. Filter out any non-transaction rows. A reliable indicator of a transaction
    # row is a valid date in the first column.
    # ICICI statements typically use DD/MM/YYYY or DD-MM-YYYY.
    date_pattern = r'^\s*\d{2}[-/]\d{2}[-/]\d{4}\s*$'
    transaction_mask = df['Date'].str.match(date_pattern, na=False)
    df = df[transaction_mask].copy()

    if df.empty:
        return pd.DataFrame(columns=['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'])

    # 4. Clean the data thoroughly:
    # - 'Date' must be a string in 'DD-MM-YYYY' format.
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.strftime('%d-%m-%Y')

    # - Remove all newlines ('\n') from the 'Description' column and replace them with a single space.
    df['Description'] = df['Description'].str.replace('\n', ' ', regex=False).str.strip()

    # - The numeric columns ('Debit Amt', 'Credit Amt', 'Balance') must be of type float.
    numeric_cols = ['Debit Amt', 'Credit Amt', 'Balance']
    for col in numeric_cols:
        # First, clean the string by removing commas.
        cleaned_series = df[col].astype(str).str.replace(',', '', regex=False).str.strip()
        # Use `pd.to_numeric` to handle conversion, coercing errors to NaN. Do not fill NaNs.
        df[col] = pd.to_numeric(cleaned_series, errors='coerce')

    # Ensure the final DataFrame has the specified columns in the correct order.
    final_df = df[['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance']]

    # Reset index for a clean, sequential index.
    final_df = final_df.reset_index(drop=True)

    return final_df