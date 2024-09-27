import os
import csv
import json
import xml.etree.ElementTree as ET
from typing import Any, List, Dict, Union
import yaml

debug_mode = False

class FileTools:
    @staticmethod
    def save_code_to_file(code: str, file_path: str):
        """
        Save the given code to a file at the specified path.

        Args:
            code (str): The code to be saved.
            file_path (str): The path where the file should be saved.

        Raises:
            OSError: If there's an error creating the directory or writing the file.
            TypeError: If the input types are incorrect.
        """
        try:
            #print(f"Attempting to save code to file: {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                file.write(code)
            print(f"\033[95mSaved code to {file_path}\033[0m")
            print(f"Successfully saved code to file: {file_path}")
        except OSError as e:
            print(f"Error creating directory or writing file at FileTools.save_code_to_file: {e}")
            print(f"OSError occurred at FileTools.save_code_to_file: {str(e)}")
        except TypeError as e:
            print(f"Invalid input type: {e}")
            print(f"TypeError occurred at FileTools.save_code_to_file: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred at FileTools.save_code_to_file: {e}")
            print(f"Unexpected error at FileTools.save_code_to_file: {str(e)}")
    
    @staticmethod
    def generate_directory_tree(base_path, additional_ignore=None):
        """
        Recursively generate a file structure dictionary for the given base path.

        Args:
            base_path (str): The root directory path to start the file structure generation.
            additional_ignore (List[str], optional): Additional files or directories to ignore.

        Returns:
            dict: A nested dictionary representing the file structure, where each directory
                is represented by a dict with 'name', 'type', and 'children' keys, and each
                file is represented by a dict with 'name', 'type', and 'contents' keys.

        Raises:
            ValueError: If the specified path is not within the current working directory.
            PermissionError: If there's a permission error accessing the directory or its contents.
            FileNotFoundError: If the specified path does not exist.
            OSError: If there's an error accessing the directory or its contents.
        """
        default_ignore_list = {".DS_Store", ".gitignore", ".env", "node_modules", "__pycache__"}
        
        if additional_ignore:
            ignore_list = default_ignore_list.union(set(additional_ignore))
        else:
            ignore_list = default_ignore_list

        #print(f"Starting file structure generation for path: {base_path}")
        #print(f"Ignore list: {ignore_list}")
        
        try:
            # Convert both paths to absolute and normalize them
            abs_base_path = os.path.abspath(os.path.normpath(base_path))
            abs_cwd = os.path.abspath(os.path.normpath(os.getcwd()))

            # Check if the base_path is within or equal to the current working directory
            if not abs_base_path.startswith(abs_cwd):
                raise ValueError(f"Access to the specified path is not allowed: {abs_base_path}")
            
            if not os.path.exists(abs_base_path):
                raise FileNotFoundError(f"The specified path does not exist: {abs_base_path}")
            
            if not os.path.isdir(abs_base_path):
                raise NotADirectoryError(f"The specified path is not a directory: {abs_base_path}")
            
            file_structure = {
                "name": os.path.basename(abs_base_path),
                "type": "directory",
                "children": []
            }

            for item in os.listdir(abs_base_path):
                if item in ignore_list or item.startswith('.'):
                    print(f"Skipping ignored or hidden item: {item}")
                    continue  # Skip ignored and hidden files/directories
                
                item_path = os.path.join(abs_base_path, item)
                print(f"Processing item: {item_path}")
                
                if os.path.isdir(item_path):
                    try:
                        file_structure["children"].append(FileTools.generate_directory_tree(item_path))
                    except PermissionError:
                        print(f"Permission denied for directory: {item_path}")
                        file_structure["children"].append({
                            "name": item,
                            "type": "directory",
                            "error": "Permission denied"
                        })
                else:
                    try:
                        with open(item_path, "r", encoding="utf-8") as file:
                            file_contents = file.read()
                            #print(f"Successfully read file contents: {item_path}")
                    except UnicodeDecodeError:
                        print(f"UTF-8 decoding failed for {item_path}, attempting ISO-8859-1")
                        try:
                            with open(item_path, "r", encoding="iso-8859-1") as file:
                                file_contents = file.read()
                                #print(f"Successfully read file contents with ISO-8859-1: {item_path}")
                        except Exception as e:
                            print(f"Failed to read file: {item_path}, Error: {str(e)}")
                            file_contents = f"Error reading file: {str(e)}"
                    except PermissionError:
                        print(f"Permission denied for file: {item_path}")
                        file_contents = "Permission denied"
                    except Exception as e:
                        print(f"Unexpected error reading file: {item_path}, Error: {str(e)}")
                        file_contents = f"Unexpected error: {str(e)}"
                    
                    file_structure["children"].append({
                        "name": item,
                        "type": "file",
                        "contents": file_contents
                    })

            print(f"Completed file structure generation for path: {abs_base_path}")
            return file_structure

        except PermissionError as e:
            print(f"Permission error accessing directory or its contents: {str(e)}")
            raise
        except FileNotFoundError as e:
            print(f"File or directory not found: {str(e)}")
            raise
        except NotADirectoryError as e:
            print(f"Not a directory error: {str(e)}")
            raise
        except OSError as e:
            print(f"OS error accessing directory or its contents: {str(e)}")
            raise
        except Exception as e:
            print(f"Unexpected error in generate_directory_tree: {str(e)}")
            raise  

    @staticmethod
    def read_file_contents(full_file_path):
        """
        Retrieve the contents of a file at the specified path.

        Args:
            full_file_path (str): The full path to the file.

        Returns:
            str: The contents of the file if successfully read, None otherwise.

        Raises:
            IOError: If there's an error reading the file.
        """
        print(f"Attempting to read file contents from: {full_file_path}")
        
        try:
            with open(full_file_path, 'r', encoding='utf-8') as file:
                file_contents = file.read()
                print("File contents successfully retrieved.")
                return file_contents
        except FileNotFoundError:
            print(f"Error: File not found at path: {full_file_path}")
            print(f"FileNotFoundError at FileTools.read_file_contents: {full_file_path}")
            return None
        except IOError as e:
            print(f"Error reading file: {e}")
            print(f"IOError while reading file at FileTools.read_file_contents: {full_file_path}. Error: {str(e)}")
            return None
        except UnicodeDecodeError:
            print(f"Error: Unable to decode file contents using UTF-8 encoding: {full_file_path}")
            print(f"UnicodeDecodeError at FileTools.read_file_contents: Attempting to read with ISO-8859-1 encoding")
            try:
                with open(full_file_path, 'r', encoding='iso-8859-1') as file:
                    file_contents = file.read()
                    #print("File contents successfully retrieved using ISO-8859-1 encoding.")
                    return file_contents
            except Exception as e:
                print(f"Error: Failed to read file with ISO-8859-1 encoding: {e}")
                #print(f"Error reading file with ISO-8859-1 encoding: {full_file_path}. Error: {str(e)}")
                return None
        except Exception as e:
            print(f"Unexpected error occurred while reading file: {e}")
            print(f"Unexpected error in FileTools.read_file_contents: {full_file_path}. Error: {str(e)}")
            return None

    @staticmethod
    def read_csv(file_path: str) -> List[Dict[str, Any]]:
        """
        Read a CSV file and return its contents as a list of dictionaries.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a row in the CSV.

        Raises:
            FileNotFoundError: If the specified file is not found.
            csv.Error: If there's an error parsing the CSV file.
        """
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                return [row for row in reader]
        except FileNotFoundError:
            print(f"Error: CSV file not found at {file_path}")
            return (f"Error: CSV file not found at {file_path}")
        except csv.Error as e:
            print(f"Error parsing CSV file: {e}")
            return (f"Error parsing CSV file: {e}")

    @staticmethod
    def read_json(file_path: str) -> Union[Dict[str, Any], List[Any]]:
        """
        Read a JSON file and return its contents.

        Args:
            file_path (str): The path to the JSON file.

        Returns:
            Union[Dict[str, Any], List[Any]]: The parsed JSON data.

        Raises:
            FileNotFoundError: If the specified file is not found.
            json.JSONDecodeError: If there's an error parsing the JSON file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as jsonfile:
                return json.load(jsonfile)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {file_path}")
            return (f"Error: JSON file not found at {file_path}")
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            return (f"Error parsing JSON file: {e}")

    @staticmethod
    def read_xml(file_path: str) -> ET.Element:
        """
        Read an XML file and return its contents as an ElementTree.

        Args:
            file_path (str): The path to the XML file.

        Returns:
            ET.Element: The root element of the parsed XML.

        Raises:
            FileNotFoundError: If the specified file is not found.
            ET.ParseError: If there's an error parsing the XML file.
        """
        try:
            tree = ET.parse(file_path)
            return tree.getroot()
        except FileNotFoundError:
            print(f"Error: XML file not found at {file_path}")
            return (f"Error: XML file not found at {file_path}")
        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")
            return (f"Error parsing XML file: {e}")

    @staticmethod
    def read_yaml(file_path: str) -> Union[Dict[str, Any], List[Any]]:
        """
        Read a YAML file and return its contents.

        Args:
            file_path (str): The path to the YAML file.

        Returns:
            Union[Dict[str, Any], List[Any]]: The parsed YAML data.

        Raises:
            FileNotFoundError: If the specified file is not found.
            yaml.YAMLError: If there's an error parsing the YAML file.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as yamlfile:
                return yaml.safe_load(yamlfile)
        except FileNotFoundError:
            print(f"Error: YAML file not found at {file_path}")
            return (f"Error: YAML file not found at {file_path}")
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return (f"Error parsing YAML file: {e}")

    @staticmethod
    def search_csv(file_path: str, search_column: str, search_value: Any) -> List[Dict[str, Any]]:
        """
        Search for a specific value in a CSV file and return matching rows.

        Args:
            file_path (str): The path to the CSV file.
            search_column (str): The name of the column to search in.
            search_value (Any): The value to search for.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing matching rows.

        Raises:
            FileNotFoundError: If the specified file is not found.
            KeyError: If the specified search column doesn't exist in the CSV.
        """
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                
                # Check if the search_column exists
                if search_column not in reader.fieldnames:
                    raise KeyError(f"Column '{search_column}' not found in the CSV file.")
                
                # Search for matching rows
                return [row for row in reader if row[search_column] == str(search_value)]
        except FileNotFoundError:
            print(f"Error: CSV file not found at {file_path}")
            return (f"Error: CSV file not found at {file_path}")
        except KeyError as e:
            print(f"Error: {e}")
            return (f"Error: {e}")

    @staticmethod
    def search_json(data: Union[Dict[str, Any], List[Any]], search_key: str, search_value: Any) -> List[Any]:
        """
        Search for a specific key-value pair in a JSON structure and return matching items.

        Args:
            data (Union[Dict[str, Any], List[Any]]): The JSON data to search.
            search_key (str): The key to search for.
            search_value (Any): The value to match.

        Returns:
            List[Any]: A list of items that match the search criteria.
        """
        results = []

        def search_recursive(item):
            if isinstance(item, dict):
                if search_key in item and item[search_key] == search_value:
                    results.append(item)
                for value in item.values():
                    search_recursive(value)
            elif isinstance(item, list):
                for element in item:
                    search_recursive(element)

        search_recursive(data)
        return results

    @staticmethod
    def search_xml(root: ET.Element, tag: str, attribute: str = None, value: str = None) -> List[ET.Element]:
        """
        Search for specific elements in an XML structure.

        Args:
            root (ET.Element): The root element of the XML to search.
            tag (str): The tag name to search for.
            attribute (str, optional): The attribute name to match. Defaults to None.
            value (str, optional): The attribute value to match. Defaults to None.

        Returns:
            List[ET.Element]: A list of matching XML elements.
        """
        if attribute and value:
            return root.findall(f".//*{tag}[@{attribute}='{value}']")
        else:
            return root.findall(f".//*{tag}")

    @staticmethod
    def search_yaml(data: Union[Dict[str, Any], List[Any]], search_key: str, search_value: Any) -> List[Any]:
        """
        Search for a specific key-value pair in a YAML structure and return matching items.

        Args:
            data (Union[Dict[str, Any], List[Any]]): The YAML data to search.
            search_key (str): The key to search for.
            search_value (Any): The value to match.

        Returns:
            List[Any]: A list of items that match the search criteria.
        """
        # YAML is parsed into Python data structures, so we can reuse the JSON search method
        return FileTools.search_json(data, search_key, search_value)

