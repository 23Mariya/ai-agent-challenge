import os
import sys
import logging
import difflib
import importlib.util
from pathlib import Path
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
try:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    GEMINI_MODEL = "models/gemini-2.5-pro"
except Exception as e:
    logging.error(f"Failed to configure Gemini API. Ensure GOOGLE_API_KEY is set. Error: {e}")
    sys.exit(1)
MAX_ATTEMPTS = 3
class PDFParserAgent:
    def __init__(self, target: str):
        self.target = target
        self.data_dir = Path(f"data/{target}")
        self.output_dir = Path("output")
        self.parser_dir = Path("custom_parsers")
        self.pdf_path = self.data_dir / f"{target}_sample.pdf"
        self.csv_path = self.data_dir / f"{target}_sample.csv"
        self.model = genai.GenerativeModel(GEMINI_MODEL)
        self.output_dir.mkdir(exist_ok=True)
        self.parser_dir.mkdir(exist_ok=True)
        (self.parser_dir / "__init__.py").touch()
    def _validate_paths(self) -> bool:
        if not self.pdf_path.exists() or not self.csv_path.exists():
            logging.error(f"Missing files. Ensure '{self.pdf_path}' and '{self.csv_path}' exist.")
            return False
        return True
    def _generate_prompt(self) -> str:
        expected_df = pd.read_csv(self.csv_path)
        expected_head = expected_df.head(5).to_string()
        return f"""
        You are a world-class Python engineer who writes flawless, production-ready code.
        Your task is to write a Python script that parses a bank statement PDF from '{self.target}' and returns a pandas DataFrame.
        The script MUST contain a single function with this exact signature: `parse(pdf_path: str) -> pd.DataFrame`
        Follow these non-negotiable rules:
        1.  Use the `camelot-py` library with `flavor='stream'` as the PDF is stream-based.
        2.  Combine tables from all pages of the PDF.
        3.  The final DataFrame MUST have these columns in this exact order: ['Date', 'Description', 'Debit Amt', 'Credit Amt', 'Balance'].
        4.  Clean the data thoroughly:
            - 'Date' must be a string in 'DD-MM-YYYY' format.
            - Remove all newlines ('\\n') from the 'Description' column and replace them with a single space.
            - The numeric columns ('Debit Amt', 'Credit Amt', 'Balance') must be of type float. Use `pd.to_numeric(column, errors='coerce')` to handle non-numeric values. Do not fill NaNs.
        5.  Filter out any non-transaction rows. A reliable indicator of a transaction row is a valid date in the first column.
        Here is the head of the target DataFrame for your reference:
        {expected_head}
        Return ONLY the raw Python code. Do not wrap it in markdown backticks or add any explanations.
        """
    def _execute_and_test_code(self, code: str) -> tuple[bool, str]:
        if not code or not code.strip():
            return False, "LLM returned empty or invalid code."
        try:
            module_name = f"custom_parsers.{self.target}_parser_{os.urandom(4).hex()}"
            spec = importlib.util.spec_from_loader(module_name, loader=None)
            module = importlib.util.module_from_spec(spec)
            exec(code, module.__dict__)
            sys.modules[module_name] = module
            actual_df = module.parse(str(self.pdf_path))
        except Exception as e:
            return False, f"Code execution failed with error: {e}"
        expected_df = pd.read_csv(self.csv_path)
        for col in ['Debit Amt', 'Credit Amt', 'Balance']:
            expected_df[col] = pd.to_numeric(expected_df[col], errors='coerce')
        actual_df.reset_index(drop=True, inplace=True)
        expected_df.reset_index(drop=True, inplace=True)
        if actual_df.equals(expected_df):
            output_path = self.output_dir / f"{self.target}_output.csv"
            actual_df.to_csv(output_path, index=False)
            return True, "✅ Success! Parser matched expected output."
        else:
            return False, "Data mismatch."
    def _save_parser(self, code: str):
        parser_path = self.parser_dir / f"{self.target}_parser.py"
        parser_path.write_text(code, encoding="utf-8")
        logging.info(f"Successfully saved final parser to {parser_path}")
    def run(self):
        if not self._validate_paths():
            return
        logging.info(f"Starting agent for target: {self.target}")
        for attempt in range(1, MAX_ATTEMPTS + 1):
            logging.info(f"--- ATTEMPT {attempt}/{MAX_ATTEMPTS} ---")
            try:
                prompt = self._generate_prompt()
                response = self.model.generate_content(prompt)
                code = response.text
                if "```python" in code:
                    code = code.split("```python\n")[1].split("```")[0]
                code = code.strip()
                is_ok, message = self._execute_and_test_code(code)
                if is_ok:
                    logging.info(message)
                    self._save_parser(code)
                    return
                else:
                    logging.warning(f"Attempt {attempt} failed: {message}")
            except Exception as e:
                logging.error(f"An unexpected error occurred in attempt {attempt}: {e}")
        logging.warning("All LLM attempts failed. No parser was saved.")
def main():
    if len(sys.argv) < 3 or sys.argv[1] != "--target":
        print("Usage: python agent_gemini.py --target <bank_name>")
        sys.exit(1)
    target_bank = sys.argv[2]
    agent = PDFParserAgent(target=target_bank)
    agent.run()
if __name__ == "__main__":
    main()