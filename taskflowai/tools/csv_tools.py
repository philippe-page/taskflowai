from typing import List, Union

def check_csv():
    try:
        import csv
    except ModuleNotFoundError:
        raise ImportError("csv is required for CSV tools. Install with `pip install csv`")
    return csv

class CSVTools:
    @staticmethod
    def read_csv(file_path: str, delimiter: str = ',') -> List[List[str]]:
        """
        Read a CSV file and return its contents as a list of lists.

        Args:
            file_path (str): The path to the CSV file.
            delimiter (str, optional): The delimiter used in the CSV file. Defaults to ','.

        Returns:
            List[List[str]]: The contents of the CSV file as a list of lists.

        Raises:
            FileNotFoundError: If the specified file is not found.
            csv.Error: If there's an error parsing the CSV file.
            IOError: If there's an error reading the file.
        """
        csv = check_csv()
        try:
            with open(file_path, 'r') as file:
                reader = csv.reader(file, delimiter=delimiter)
                data = [row for row in reader]
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{file_path}' not found.")
        except csv.Error as e:
            raise csv.Error(f"Error parsing CSV file: {e}")
        except IOError as e:
            raise IOError(f"Error reading file: {e}")

    @staticmethod
    def write_csv(file_path: str, data: List[List[str]], delimiter: str = ',') -> Union[bool, str]:
        """
        Write data to a CSV file.

        Args:
            file_path (str): The path to the CSV file.
            data (List[List[str]]): The data to write to the CSV file.
            delimiter (str, optional): The delimiter to use in the CSV file. Defaults to ','.

        Returns:
            Union[bool, str]: True if the data was successfully written, or an error message as a string.
        """
        csv = check_csv()
        try:
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file, delimiter=delimiter)
                writer.writerows(data)
            return f"Successfully wrote CSV file to {file_path}."
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while writing the CSV file: {e}"
            print(error_msg)
            return error_msg

    @staticmethod
    def get_column(data: List[List[str]], column_index: int) -> Union[List[str], str]:
        """
        Extract a specific column from a list of lists representing CSV data.

        Args:
            data (List[List[str]]): The CSV data as a list of lists.
            column_index (int): The index of the column to extract (0-based).

        Returns:
            Union[List[str], str]: The extracted column as a list of strings, or an error message as a string.
        """
        csv = check_csv()
        try:
            if not data:
                error_msg = "Error: Input data is empty."
                print(error_msg)
                return error_msg
            
            num_columns = len(data[0])
            if column_index < 0 or column_index >= num_columns:
                error_msg = f"Error: Invalid column index. Must be between 0 and {num_columns - 1}."
                print(error_msg)
                return error_msg

            column = [row[column_index] for row in data]
            return column
        except IndexError:
            error_msg = "Error: Inconsistent number of columns in the input data."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while extracting the column: {e}"
            print(error_msg)
            return error_msg

    @staticmethod
    def filter_rows(data: List[List[str]], column_index: int, value: str) -> Union[List[List[str]], str]:
        """
        Filter rows in a list of lists representing CSV data based on a specific column value.

        Args:
            data (List[List[str]]): The CSV data as a list of lists.
            column_index (int): The index of the column to filter on (0-based).
            value (str): The value to match in the specified column.

        Returns:
            Union[List[List[str]], str]: The filtered rows as a list of lists, or an error message as a string.
        """
        csv = check_csv()
        try:
            if not data:
                error_msg = "Error: Input data is empty."
                print(error_msg)
                return error_msg
            
            num_columns = len(data[0])
            if column_index < 0 or column_index >= num_columns:
                error_msg = f"Error: Invalid column index. Must be between 0 and {num_columns - 1}."
                print(error_msg)
                return error_msg

            filtered_rows = [row for row in data if row[column_index] == value]
            return filtered_rows
        except IndexError:
            error_msg = "Error: Inconsistent number of columns in the input data."
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while filtering rows: {e}"
            print(error_msg)
            return error_msg

    @staticmethod
    def peek_csv(file_path: str, num_lines: int = 5) -> Union[List[List[str]], str]:
        """
        Peek at the first few lines of a CSV file.

        Args:
            file_path (str): The path to the CSV file.
            num_lines (int, optional): The number of lines to peek. Defaults to 5.

        Returns:
            Union[List[List[str]], str]: The first few lines of the CSV as a list of lists, or an error message as a string.
        """
        csv = check_csv()
        try:
            with open(file_path, 'r', newline='') as csvfile:
                csv_reader = csv.reader(csvfile)
                peeked_data = [next(csv_reader) for _ in range(num_lines)]
            return peeked_data
        except FileNotFoundError:
            error_msg = f"Error: File not found at {file_path}"
            print(error_msg)
            return error_msg
        except csv.Error as e:
            error_msg = f"Error: CSV parsing error - {str(e)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"Error: An unexpected error occurred while peeking at the CSV: {str(e)}"
            print(error_msg)
            return error_msg