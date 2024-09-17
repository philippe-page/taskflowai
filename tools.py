import csv
import base64
import json
import os
import random
import time
import xml.etree.ElementTree as ET
from datetime import timedelta, datetime
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, Callable, Set, Type 
import pandas as pd
from bs4 import BeautifulSoup
import cohere
from langchain_core.tools import BaseTool
from langchain_community.tools import _module_lookup
from fake_useragent import UserAgent
from typing import Any, Callable, Set, Type, List
from langchain_core.tools import BaseTool
from pydantic.v1 import BaseModel
import json
import requests
import yaml
from dotenv import load_dotenv

debug_mode = False

def print_debug(message):
    if not debug_mode:
        return
    print("\033[93m" + message + "\033[0m")  # Yellow

def print_error(message):
    print("\033[91m" + message + "\033[0m")  # Red

class WebTools:
    @staticmethod
    def exa_search(queries: Union[str, List[str]], num_results: int = 10, search_type: str = "neural", num_sentences: int = 5, highlights_per_url: int = 3) -> dict:
        """
        Searches the internet using the Exa search engine and returns a structured response.

        This function sends one or more search queries to the Exa API, processes the responses,
        and returns a structured dictionary containing the search results.

        Args:
            queries (Union[str, List[str]]): The search query string or a list of query strings.
            num_results (int, optional): The number of search results to return per query. Defaults to 3.
            search_type (str, optional): The type of search to perform. Can be 'neural' or 'keyword'. Defaults to 'neural'.
            num_sentences (int, optional): The number of sentences to include in each highlight. Defaults to 3.
            highlights_per_url (int, optional): The number of highlights to include per URL. Defaults to 3.

        Returns:
            dict: A structured dictionary containing the search results. The dictionary includes:
                - 'queries': The original search query or queries.
                - 'results': A list of dictionaries, each containing:
                    - 'query': The specific query for this result set.
                    - 'data': A list of dictionaries, each containing:
                        - 'title': The title of the search result.
                        - 'url': The URL of the search result.
                        - 'author': The author of the content (if available).
                        - 'highlights': A string of highlighted text snippets from the search result.

        Raises:
            requests.exceptions.RequestException: If there's an error with the API request.
            json.JSONDecodeError: If the API response cannot be parsed as JSON.

        Note:
            This function requires the EXA_API_KEY environment variable to be set.
        """
        print(f"Searching Exa for: '{queries}'")
        
        # Load environment variables from .env file
        load_dotenv()
        
        # Retrieve the API key from environment variables
        api_key = os.getenv("EXA_API_KEY")
        if not api_key:
            raise ValueError("EXA_API_KEY environment variable is not set")
        
        # Convert single query to list for consistent processing
        if isinstance(queries, str):
            queries = [queries]
        
        # Define the API endpoint and headers
        url = "https://api.exa.ai/search"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": api_key  # Use the retrieved API key
        }

        structured_data = {
            "queries": queries,
            "results": []
        }

        for query in queries:
            # Define the payload
            payload = {
                "query": query,
                "type": search_type,
                "use_autoprompt": False,
                "num_results": num_results,
                "contents": {
                    "highlights": {
                        "numSentences": num_sentences,
                        "highlightsPerUrl": highlights_per_url,
                        "query": query
                    }
                }
            }

            # Attempt to make the POST request with retries
            for attempt in range(3):  # Retry up to 3 times
                try:
                    response = requests.post(url, headers=headers, json=payload)
                    response.raise_for_status()  # Will raise an exception for HTTP errors
                    # Print the raw response
                    print_debug(f"DEBUG: Raw API response for query '{query}': {response.text}")
                    # Restructure and clean up the response data
                    data = response.json()
                    query_results = {
                        "query": query,
                        "data": []
                    }
                    for result in data["results"]:
                        structured_result = {
                            "title": result["title"],
                            "url": result["url"],
                            "author": result["author"],
                            "highlights": "\n".join(result["highlights"])
                        }
                        query_results["data"].append(structured_result)
                    
                    structured_data["results"].append(query_results)
                    break  # Success, break the retry loop
                
                except requests.exceptions.HTTPError as e:
                    print(f"HTTP error occurred for query '{query}': {e}")
                except requests.exceptions.RequestException as e:
                    print(f"Error during request to Exa for query '{query}': {e}")
                except ValueError as e:
                    print(f"Error decoding JSON for query '{query}': {e}")

                if attempt < 2:  # Don't sleep after the last attempt
                    print(f"Retrying query '{query}'...")
                    time.sleep(1)  # Wait for 1 second before retrying

            else:  # This else clause is executed if the for loop completes without breaking
                print(f"All attempts failed for query '{query}'. Adding empty result.")
                structured_data["results"].append({"query": query, "data": []})

        return structured_data


    @staticmethod
    def scrape_urls(urls: Union[str, List[str]], include_html: bool = False, include_links: bool = False) -> List[Dict[str, Any]]:
        if isinstance(urls, str):
            urls = [urls]
        
        if not isinstance(include_html, bool) or not isinstance(include_links, bool):
            raise ValueError("include_html and include_links must be boolean values")

        results = []
        ua = UserAgent()

        for url in urls:
            try:
                time.sleep(random.uniform(.1, .3))
                headers = {'User-Agent': ua.random}
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                response.encoding = response.apparent_encoding

                soup = BeautifulSoup(response.text, 'html.parser')

                # Remove common non-content elements
                for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                    element.decompose()

                # Try to find the main content
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                
                if main_content:
                    content_soup = main_content
                else:
                    # If no main content found, use the whole body but remove potential sidebars
                    content_soup = soup.find('body')
                    if content_soup:
                        for sidebar in content_soup(['aside', 'div'], class_=['sidebar', 'widget']):
                            sidebar.decompose()

                result: Dict[str, Any] = {'url': url}

                if include_html:
                    result['content'] = str(content_soup) if content_soup else ''
                else:
                    result['content'] = content_soup.get_text(separator=' ', strip=True) if content_soup else ''

                if include_links:
                    result['links'] = [a['href'] for a in (content_soup.find_all('a', href=True) if content_soup else [])]

                results.append(result)

            except requests.RequestException as e:
                results.append({'url': url, 'error': f"Error fetching {url}: {str(e)}"})
            except UnicodeDecodeError as e:
                results.append({'url': url, 'error': f"Encoding error for {url}: {str(e)}"})
            except (AttributeError, ValueError, TypeError) as e:
                results.append({'url': url, 'error': f"Error processing {url}: {str(e)}"})

        return results

    @staticmethod
    def serper_search(
        query: Union[str, List[str]],
        search_type: Literal["search", "news", "images", "shopping"] = "search",
        num_results: Optional[int] = range(3, 10),
        date_range: Optional[Literal["h", "d", "w", "m", "y"]] = None,
        location: Optional[str] = None
    ) -> Union[str, List[str]]:
        """
        Perform a search using the Serper API and format the results.

        Args:
            query (Union[str, List[str]]): The search query or a list of search queries.
            search_type (Literal["search", "news", "images", "shopping"]): The type of search to perform.
            num_results (Optional[int]): Number of results to return (default range: 3-10).
            date_range (Optional[Literal["h", "d", "w", "m", "y"]]): Date range for results (h: hour, d: day, w: week, m: month, y: year).
            location (Optional[str]): Specific location for the search.

        Returns:
            Union[str, List[str]]: A formatted string or list of formatted strings containing the search results.

        Raises:
            ValueError: If the API key is not set or if there's an error with the API call.
            requests.RequestException: If there's an error with the HTTP request.
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("SERPER_API_KEY environment variable is not set")

        # Define the base URL
        BASE_URL = "https://google.serper.dev"

        # Construct the URL based on the search type
        url = f"{BASE_URL}/{search_type}"

        # Convert query to list if it's a string
        queries = [query] if isinstance(query, str) else query

        results_list = []

        for single_query in queries:
            # Prepare the payload
            payload = {
                "q": single_query,
                "gl": "us",
                "hl": "en",
            }

            # Add num_results to payload if provided
            if num_results is not None:
                payload["num"] = num_results

            # Add optional parameters if provided
            if date_range:
                payload["tbs"] = f"qdr:{date_range}"
            if location:
                payload["location"] = location

            # Prepare headers
            headers = {
                "X-API-KEY": api_key,
                "Content-Type": "application/json"
            }

            try:
                # Make the API call
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                response.raise_for_status()  # Raise an exception for HTTP errors
                
                # Parse the JSON response
                results = response.json()
                
                # Format the results
                formatted_results = ""

                if "organic" in results:
                    formatted_results += "Organic Results:\n"
                    for i, result in enumerate(results["organic"], 1):
                        formatted_results += f"{i}. {result.get('title', 'No Title')}\n"
                        formatted_results += f"   URL: {result.get('link', 'No Link')}\n"
                        formatted_results += f"   Snippet: {result.get('snippet', 'No Snippet')}\n\n"

                if "news" in results:
                    formatted_results += "News Results:\n"
                    for i, news in enumerate(results["news"], 1):
                        formatted_results += f"{i}. {news.get('title', 'No Title')}\n"
                        formatted_results += f"   Source: {news.get('source', 'No Source')}\n"
                        formatted_results += f"   URL: {news.get('link', 'No Link')}\n"
                        formatted_results += f"   Date: {news.get('date', 'No Date')}\n"
                        formatted_results += f"   Snippet: {news.get('snippet', 'No Snippet')}\n"
                        formatted_results += f"   Image URL: {news.get('imageUrl', 'No Image URL')}\n\n"

                if "images" in results:
                    formatted_results += "Image Results:\n"
                    for i, image in enumerate(results["images"], 1):
                        formatted_results += f"{i}. {image.get('title', 'No Title')}\n"
                        formatted_results += f"   URL: {image.get('link', 'No Link')}\n"
                        formatted_results += f"   Source: {image.get('source', 'No Source')}\n\n"

                if "shopping" in results:
                    formatted_results += "Shopping Results:\n"
                    for i, item in enumerate(results["shopping"], 1):
                        formatted_results += f"{i}. {item.get('title', 'No Title')}\n"
                        formatted_results += f"   Price: {item.get('price', 'No Price')}\n"
                        formatted_results += f"   URL: {item.get('link', 'No Link')}\n\n"

                results_list.append(formatted_results.strip())
            
            except requests.RequestException as e:
                results_list.append(f"Error making request to Serper API for query '{single_query}': {str(e)}")
            except json.JSONDecodeError:
                results_list.append(f"Error decoding JSON response from Serper API for query '{single_query}'")

        return results_list[0] if len(results_list) == 1 else results_list

    @staticmethod
    def scrape_url_with_serper(urls: Union[str, List[str]]) -> Union[dict, List[dict]]:
        """
        Scrape one or more webpages using the Serper API.

        Args:
            urls (Union[str, List[str]]): A single URL string or a list of URL strings to scrape.

        Returns:
            Union[dict, List[dict]]: A single JSON response or a list of JSON responses from the Serper API.

        Raises:
            ValueError: If the API key is not set or if the input is neither a string nor a list.
            requests.RequestException: If there's an error with the API request.
            json.JSONDecodeError: If the API response cannot be parsed as JSON.
        """
        print_debug(f"Attempting to scrape URL(s): {urls}")
        
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise ValueError("SERPER_API_KEY environment variable is not set")

        serper_url = "https://scrape.serper.dev"
        headers = {
            'X-API-KEY': api_key,
            'Content-Type': 'application/json'
        }

        def scrape_single_url(url: str) -> dict:
            payload = json.dumps({"url": url})
            try:
                response = requests.post(serper_url, headers=headers, data=payload)
                response.raise_for_status()
                scraped_data = response.json()
                print_debug(f"Successfully scraped URL: {url}")
                return scraped_data
            except requests.RequestException as e:
                print_error(f"Error making request to Serper API for URL {url}: {str(e)}")
                raise
            except json.JSONDecodeError as e:
                print_error(f"Error decoding JSON response from Serper API for URL {url}: {str(e)}")
                raise

        if isinstance(urls, str):
            return scrape_single_url(urls)
        elif isinstance(urls, list):
            return [scrape_single_url(url) for url in urls]
        else:
            raise ValueError("Input must be a string (single URL) or a list of strings (multiple URLs)")

    @staticmethod
    def query_arxiv_api(
        query: str,
        max_results: int = 10,
        sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = "relevance",
        include_abstract: bool = True
    ) -> dict:
        """
        Query the Arxiv API and return simplified results for each entry.

        Args:
            query (str): Search query string.
            max_results (int): The maximum number of results to return. Default is 10.
            sort_by (Literal["relevance", "lastUpdatedDate", "submittedDate"]): The sorting criteria. Default is "relevance".
            include_abstract (bool): Whether to include the abstract in the response. Default is True.

        Returns:
            dict: A structured dictionary containing the simplified query results.

        Raises:
            ValueError: If there's an error with the API call or response parsing.
            requests.RequestException: If there's an error with the HTTP request.
        """
        base_url = "http://export.arxiv.org/api/query"
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            feed = BeautifulSoup(response.text, "lxml-xml")
            entries = feed.find_all("entry")

            results = {
                "total_results": int(feed.find("opensearch:totalResults").text),
                "entries": []
            }

            for entry in entries:
                entry_data = {
                    "id": entry.find("id").text,
                    "title": entry.find("title").text,
                    "authors": [author.find("name").text for author in entry.find_all("author")],
                    "published": entry.find("published").text,
                    "updated": entry.find("updated").text,
                    "categories": [category["term"] for category in entry.find_all("category")],
                    "primary_category": entry.find("arxiv:primary_category")["term"],
                    "link": entry.find("link", attrs={"type": "text/html"})["href"]
                }

                if include_abstract:
                    entry_data["abstract"] = entry.find("summary").text

                results["entries"].append(entry_data)

            return results

        except (ValueError, requests.exceptions.RequestException) as e:
            print_error(f"Error querying Arxiv API: {e}")
            raise

    @staticmethod
    def get_weather_data(
        location: str,
        forecast_days: Optional[int] = None,
        include_current: bool = True,
        include_forecast: bool = True,  # New parameter
        include_astro: bool = False,
        include_hourly: bool = False,
        include_alerts: bool = False,
    ) -> str:
        """
        Get a detailed weather report for a location in a readable format.

        Args:
            location (str): The location to get the weather report for.
            forecast_days (Optional[int]): Number of days for the forecast (1-10). If provided, forecast is included.
            include_current (bool): Whether to include current conditions in the output.
            include_forecast (bool): Whether to include forecast data in the output.
            include_astro (bool): Whether to include astronomical data in the output.
            include_hourly (bool): Whether to include hourly forecast data in the output.
            include_alerts (bool): Whether to include weather alerts in the output.

        Returns:
            str: Formatted string with the requested weather data.

        Raises:
            ValueError: If the API key is not set, days are out of range, or if the API returns an error.
            requests.RequestException: If there's an error with the API request.
        """
        load_dotenv()
        api_key = os.getenv("WEATHER_API_KEY")
        if not api_key:
            raise ValueError("WEATHER_API_KEY environment variable is not set")

        BASE_URL = "http://api.weatherapi.com/v1"
        endpoint = 'forecast.json'

        params = {
            'key': api_key,
            'q': location,
            'aqi': 'yes',
            'alerts': 'yes',
        }

        if forecast_days is not None:
            if not 1 <= forecast_days <= 10:
                raise ValueError("forecast_days must be between 1 and 10")
            params['forecast_days'] = forecast_days
        elif include_forecast:
            # If include_forecast is True but forecast_days is not set, default to 1 day
            params['forecast_days'] = 1

        try:
            response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'error' in data:
                raise ValueError(f"API Error: {data['error']['message']}")
        except requests.RequestException as e:
            raise requests.RequestException(f"Error making request to WeatherAPI: {str(e)}")

        report = "Forecast:\n"
        for key, value in data['location'].items():
            report += f"{key}: {value}\n"

        if include_current:
            report += "\nCurrent Conditions:\n"
            for key, value in data['current'].items():
                if isinstance(value, dict):
                    report += f"{key}:\n"
                    for sub_key, sub_value in value.items():
                        report += f"  {sub_key}: {sub_value}\n"
                else:
                    report += f"{key}: {value}\n"

            if include_astro:
                report += "Current Astro:\n"
                for key, value in data['forecast']['forecastday'][0]['astro'].items():
                    report += f"  {key}: {value}\n"

        if include_forecast and 'forecast' in data:
            report += "\nForecast:\n"
            for day in data['forecast']['forecastday']:
                report += f"\nDate: {day['date']}\n"
                for key, value in day['day'].items():
                    if isinstance(value, dict):
                        report += f"{key}:\n"
                        for sub_key, sub_value in value.items():
                            report += f"  {sub_key}: {sub_value}\n"
                    else:
                        report += f"{key}: {value}\n"

                if include_astro:
                    report += "Astro:\n"
                    for key, value in day['astro'].items():
                        report += f"  {key}: {value}\n"

                if include_hourly:
                    report += "Hour by hour:\n"
                    for hour in day['hour']:
                        report += f"  Time: {hour['time']}\n"
                        for key, value in hour.items():
                            if key != 'time':
                                if isinstance(value, dict):
                                    report += f"    {key}:\n"
                                    for sub_key, sub_value in value.items():
                                        report += f"      {sub_key}: {sub_value}\n"
                                else:
                                    report += f"    {key}: {value}\n"

        if include_alerts and 'alerts' in data and 'alert' in data['alerts']:
            report += "\nWeather Alerts:\n"
            for alert in data['alerts']['alert']:
                for key, value in alert.items():
                    report += f"{key}: {value}\n"
                report += "\n"

        return report


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
            print_debug(f"Attempting to save code to file: {file_path}")
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as file:
                file.write(code)
            print(f"\033[95mSaved code to {file_path}\033[0m")
            print_debug(f"Successfully saved code to file: {file_path}")
        except OSError as e:
            print(f"Error creating directory or writing file: {e}")
            print_debug(f"OSError occurred: {str(e)}")
        except TypeError as e:
            print(f"Invalid input type: {e}")
            print_debug(f"TypeError occurred: {str(e)}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            print_debug(f"Unexpected error: {str(e)}")
    
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

        print_debug(f"Starting file structure generation for path: {base_path}")
        print_debug(f"Ignore list: {ignore_list}")
        
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
                    print_debug(f"Skipping ignored or hidden item: {item}")
                    continue  # Skip ignored and hidden files/directories
                
                item_path = os.path.join(abs_base_path, item)
                print_debug(f"Processing item: {item_path}")
                
                if os.path.isdir(item_path):
                    try:
                        file_structure["children"].append(FileTools.generate_directory_tree(item_path))
                    except PermissionError:
                        print_debug(f"Permission denied for directory: {item_path}")
                        file_structure["children"].append({
                            "name": item,
                            "type": "directory",
                            "error": "Permission denied"
                        })
                else:
                    try:
                        with open(item_path, "r", encoding="utf-8") as file:
                            file_contents = file.read()
                            print_debug(f"Successfully read file contents: {item_path}")
                    except UnicodeDecodeError:
                        print_debug(f"UTF-8 decoding failed for {item_path}, attempting ISO-8859-1")
                        try:
                            with open(item_path, "r", encoding="iso-8859-1") as file:
                                file_contents = file.read()
                                print_debug(f"Successfully read file contents with ISO-8859-1: {item_path}")
                        except Exception as e:
                            print_debug(f"Failed to read file: {item_path}, Error: {str(e)}")
                            file_contents = f"Error reading file: {str(e)}"
                    except PermissionError:
                        print_debug(f"Permission denied for file: {item_path}")
                        file_contents = "Permission denied"
                    except Exception as e:
                        print_debug(f"Unexpected error reading file: {item_path}, Error: {str(e)}")
                        file_contents = f"Unexpected error: {str(e)}"
                    
                    file_structure["children"].append({
                        "name": item,
                        "type": "file",
                        "contents": file_contents
                    })

            print_debug(f"Completed file structure generation for path: {abs_base_path}")
            return file_structure

        except PermissionError as e:
            print_error(f"Permission error accessing directory or its contents: {str(e)}")
            raise
        except FileNotFoundError as e:
            print_error(f"File or directory not found: {str(e)}")
            raise
        except NotADirectoryError as e:
            print_error(f"Not a directory error: {str(e)}")
            raise
        except OSError as e:
            print_error(f"OS error accessing directory or its contents: {str(e)}")
            raise
        except Exception as e:
            print_error(f"Unexpected error in generate_directory_tree: {str(e)}")
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
        print_debug(f"Attempting to read file contents from: {full_file_path}")
        
        try:
            with open(full_file_path, 'r', encoding='utf-8') as file:
                file_contents = file.read()
                print_debug("File contents successfully retrieved.")
                return file_contents
        except FileNotFoundError:
            print(f"Error: File not found at path: {full_file_path}")
            print_debug(f"FileNotFoundError: {full_file_path}")
            return None
        except IOError as e:
            print(f"Error reading file: {e}")
            print_debug(f"IOError while reading file: {full_file_path}. Error: {str(e)}")
            return None
        except UnicodeDecodeError:
            print(f"Error: Unable to decode file contents using UTF-8 encoding: {full_file_path}")
            print_debug(f"UnicodeDecodeError: Attempting to read with ISO-8859-1 encoding")
            try:
                with open(full_file_path, 'r', encoding='iso-8859-1') as file:
                    file_contents = file.read()
                    print_debug("File contents successfully retrieved using ISO-8859-1 encoding.")
                    return file_contents
            except Exception as e:
                print(f"Error: Failed to read file with ISO-8859-1 encoding: {e}")
                print_debug(f"Error reading file with ISO-8859-1 encoding: {full_file_path}. Error: {str(e)}")
                return None
        except Exception as e:
            print(f"Unexpected error occurred while reading file: {e}")
            print_debug(f"Unexpected error in read_file_contents: {full_file_path}. Error: {str(e)}")
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
            raise
        except csv.Error as e:
            print(f"Error parsing CSV file: {e}")
            raise

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
            raise
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON file: {e}")
            raise

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
            raise
        except ET.ParseError as e:
            print(f"Error parsing XML file: {e}")
            raise

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
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            raise

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
            df = pd.read_csv(file_path)
            if search_column not in df.columns:
                raise KeyError(f"Column '{search_column}' not found in the CSV file.")
            return df[df[search_column] == search_value].to_dict('records')
        except FileNotFoundError:
            print(f"Error: CSV file not found at {file_path}")
            raise
        except KeyError as e:
            print(f"Error: {e}")
            raise

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


class EmbeddingsTools:
    
    MODEL_DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
        "embed-english-v3.0": 1024,
        "mistral-embed": 1024
    }

    @staticmethod
    def get_openai_embeddings(
        input_text: Union[str, List[str]],
        model: Literal["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"] = "text-embedding-3-small"
    ) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Generate embeddings for the given input text using OpenAI's API.

        Args:
            input_text (Union[str, List[str]]): The input text or list of texts to embed.
            model (Literal["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]): 
                The model to use for generating embeddings. Default is "text-embedding-3-small".

        Returns:
            Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
                - A list of embeddings.
                - A dictionary with the number of dimensions for the chosen model.

        Raises:
            ValueError: If the API key is not set or if there's an error with the API call.
            requests.exceptions.RequestException: If there's an error with the HTTP request.

        Note:
            This method requires a valid OpenAI API key to be set in the OPENAI_API_KEY environment variable.
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Prepare the API request
        url = "https://api.openai.com/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Ensure input_text is a list
        if isinstance(input_text, str):
            input_text = [input_text]

        payload = {
            "input": input_text,
            "model": model,
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            embeddings = [item['embedding'] for item in data['data']]

            return embeddings, {"dimensions": EmbeddingsTools.MODEL_DIMENSIONS[model]}

        except requests.exceptions.RequestException as e:
            raise requests.exceptions.RequestException(f"Error making request to OpenAI API: {str(e)}")

    @staticmethod
    def get_cohere_embeddings(
        input_text: Union[str, List[str]],
        model: str = "embed-english-v3.0",
        input_type: str = "search_document"
    ) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Generate embeddings for the given input text using Cohere's API.

        Args:
            input_text (Union[str, List[str]]): The input text or list of texts to embed.
            model (str): The model to use for generating embeddings. Default is "embed-english-v3.0".
            input_type (str): The type of input. Default is "search_document".

        Returns:
            Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
                - A list of embeddings.
                - A dictionary with the number of dimensions for the chosen model.

        Raises:
            ValueError: If the API key is not set.
            RuntimeError: If there's an error in generating embeddings from the Cohere API.
        """
        load_dotenv()
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY environment variable is not set")

        cohere_client = cohere.Client(api_key)

        if isinstance(input_text, str):
            input_text = [input_text]

        try:
            time.sleep(1)  # Rate limiting
            response = cohere_client.embed(
                texts=input_text,
                model=model,
                input_type=input_type
            )
            embeddings = response.embeddings
            return embeddings, {"dimensions": EmbeddingsTools.MODEL_DIMENSIONS[model]}

        except Exception as e:
            raise RuntimeError(f"Failed to get embeddings from Cohere API: {str(e)}")

    @staticmethod
    def get_mistral_embeddings(
        input_text: Union[str, List[str]],
        model: str = "mistral-embed"
    ) -> Tuple[List[List[float]], Dict[str, int]]:
        """
        Generate embeddings for the given input text using Mistral AI's API.

        Args:
            input_text (Union[str, List[str]]): The input text or list of texts to embed.
            model (str): The model to use for generating embeddings. Default is "mistral-embed".

        Returns:
            Tuple[List[List[float]], Dict[str, int]]: A tuple containing:
                - A list of embeddings.
                - A dictionary with the number of dimensions for the chosen model.

        Raises:
            ValueError: If the API key is not set or if there's an error with the API call.
            requests.exceptions.RequestException: If there's an error with the HTTP request.

        Note:
            This method requires a valid Mistral AI API key to be set in the MISTRAL_API_KEY environment variable.
        """
        # Load environment variables
        load_dotenv()
        
        # Get API key from environment variable
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is not set")

        # Prepare the API request
        url = "https://api.mistral.ai/v1/embeddings"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # Ensure input_text is a list
        if isinstance(input_text, str):
            input_text = [input_text]

        payload = {
            "model": model,
            "input": input_text
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            embeddings = [item['embedding'] for item in data['data']]
            dimensions = len(embeddings[0]) if embeddings else 0

            return embeddings, {"dimensions": dimensions}

        except requests.exceptions.RequestException as e:
            if hasattr(e.response, 'text'):
                error_details = e.response.text
            else:
                error_details = str(e)
            raise requests.exceptions.RequestException(f"Error making request to Mistral AI API: {error_details}")


class WikipediaTools:
    @staticmethod
    def get_article(title, include_images=False):
        """
        Retrieve a Wikipedia article by its title.

        This method fetches the content of a Wikipedia article, including its extract, URL, and optionally, images.

        Args:
            title (str): The title of the Wikipedia article to retrieve.
            include_images (bool, optional): Whether to include images in the response. Defaults to False.

        Returns:
            dict: A dictionary containing the article data, including:
                - 'extract': The main text content of the article.
                - 'fullurl': The full URL of the article on Wikipedia.
                - 'pageid': The unique identifier of the page.
                - 'title': The title of the article.
                - 'thumbnail' (optional): Information about the article's thumbnail image, if available and requested.

        Raises:
            requests.exceptions.RequestException: If there's an error fetching the article from the Wikipedia API.
            KeyError, ValueError: If there's an error parsing the API response.

        Note:
            This method uses the Wikipedia API to fetch article data. The API has rate limits and usage policies
            that should be respected when making frequent requests.
        """
        print_debug(f"Getting article for title: {title}")
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "extracts|info|pageimages",
            "inprop": "url",
            "redirects": "",
            "format": "json",
            "origin": "*",
            "pithumbsize": "400" if include_images else "0"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            page_id = next(iter(pages))
            article = pages[page_id]
            return article
        except requests.exceptions.RequestException as e:
            print_error(f"Error fetching article: {e}")
            return None
        except (KeyError, ValueError) as e:
            print_error(f"Error parsing response: {e}")
            return None

    @staticmethod
    def search_articles(query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Search for Wikipedia articles based on a given query.

        Args:
            query (str): The search query string.
            num_results (int, optional): The maximum number of search results to return. Defaults to 10.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing detailed information about each search result.
            Each dictionary includes:
                - 'title': The title of the article.
                - 'fullurl': The full URL of the article on Wikipedia.
                - 'snippet': A brief extract or snippet from the article.

        Raises:
            requests.exceptions.RequestException: If there's an error fetching search results from the Wikipedia API.
            KeyError, ValueError: If there's an error parsing the API response.

        Note:
            This method uses the Wikipedia API to perform the search. The API has rate limits and usage policies
            that should be respected when making frequent requests.
        """
        print_debug(f"Searching articles for query: {query}")
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": num_results,
            "format": "json",
            "origin": "*"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            search_results = data["query"]["search"]

            # Fetch additional details for each search result
            detailed_results = []
            for result in search_results:
                page_id = result['pageid']
                detailed_params = {
                    "action": "query",
                    "pageids": page_id,
                    "prop": "info|extracts|pageimages",
                    "inprop": "url",
                    "exintro": "",
                    "explaintext": "",
                    "pithumbsize": "250",
                    "format": "json",
                    "origin": "*"
                }
                detailed_response = requests.get(base_url, params=detailed_params)
                detailed_response.raise_for_status()
                detailed_data = detailed_response.json()
                page_data = detailed_data["query"]["pages"][str(page_id)]
                
                detailed_result = {
                    "title": page_data.get("title"),
                    "fullurl": page_data.get("fullurl"),
                    "snippet": page_data.get("extract", "")
                }
                detailed_results.append(detailed_result)

            return detailed_results
        except requests.exceptions.RequestException as e:
            print_error(f"Error searching articles: {e}")
            return []
        except (KeyError, ValueError) as e:
            print_error(f"Error parsing response: {e}")
            return []

    @staticmethod
    def get_main_image(title: str, thumb_size: int = 250) -> Optional[str]:
        """
        Retrieve the main image for a given Wikipedia article title.

        This method queries the Wikipedia API to fetch the main image (thumbnail) 
        associated with the specified article title.

        Args:
            title (str): The title of the Wikipedia article.
            thumb_size (int, optional): The desired size of the thumbnail in pixels. Defaults to 250.

        Returns:
            Optional[str]: The URL of the main image if found, None otherwise.

        Raises:
            requests.exceptions.RequestException: If there's an error in the HTTP request.
            KeyError, ValueError: If there's an error parsing the API response.
        """
        print_debug(f"Getting main image for title: {title}")
        base_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageimages",
            "pithumbsize": thumb_size,
            "format": "json",
            "origin": "*"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            page_id = next(iter(pages))
            image_info = pages[page_id].get("thumbnail")
            if image_info:
                return image_info["source"]
            else:
                return None
        except requests.exceptions.RequestException as e:
            print_error(f"Error fetching main image: {e}")
            return None
        except (KeyError, ValueError) as e:
            print_error(f"Error parsing response: {e}")
            return None

    @staticmethod
    def search_images(query: str, limit: int = 20, thumb_size: int = 250) -> List[Dict[str, str]]:
        """
        Search for images on Wikimedia Commons based on a given query.

        This method queries the Wikimedia Commons API to fetch images related to the specified query.

        Args:
            query (str): The search query for finding images.
            limit (int, optional): The maximum number of image results to return. Defaults to 20.
            thumb_size (int, optional): The desired size of the thumbnail in pixels. Defaults to 250.

        Returns:
            List[Dict[str, str]]: A list of dictionaries containing image information.
            Each dictionary includes 'title', 'url', and 'thumbnail' keys.

        Raises:
            requests.exceptions.RequestException: If there's an error in the HTTP request.
            KeyError, ValueError: If there's an error parsing the API response.
        """
        print_debug(f"Searching images for query: {query}")
        base_url = "https://commons.wikimedia.org/w/api.php"
        params = {
            "action": "query",
            "generator": "search",
            "gsrnamespace": "6",
            "gsrsearch": f"intitle:{query}",
            "gsrlimit": limit,
            "prop": "pageimages|info",
            "pithumbsize": thumb_size,
            "inprop": "url",
            "format": "json",
            "origin": "*"
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            pages = data["query"]["pages"]
            image_results = []
            for page_id, page_data in pages.items():
                image_info = {
                    "title": page_data["title"],
                    "url": page_data["fullurl"],
                    "thumbnail": page_data.get("thumbnail", {}).get("source")
                }
                image_results.append(image_info)

            # New code to format the output
            formatted_results = []
            separator = "-" * 30  # Create a separator line
            for i, image in enumerate(image_results, 1):
                formatted_image = f"\nImage {i}:\n"
                formatted_image += f"  Title: {image['title']}\n"
                formatted_image += f"  URL: {image['url']}\n"
                formatted_image += f"  Thumbnail: {image['thumbnail']}\n"
                formatted_image += f"{separator}"
                formatted_results.append(formatted_image)

            return "".join(formatted_results)

        except requests.exceptions.RequestException as e:
            print_error(f"Error searching images: {e}")
            return "Error occurred while searching for images."
        except (KeyError, ValueError) as e:
            print_error(f"Error parsing response: {e}")
            return "Error occurred while parsing the response."


class GitHubTools:

    @staticmethod
    def get_user_info(username: str) -> Dict[str, Any]:
        """
        Get public information about a GitHub user.

        Args:
            username (str): The GitHub username of the user.

        Returns:
            Dict[str, Any]: A dictionary containing the user's public information.
                            Keys may include 'login', 'id', 'name', 'company', 'blog',
                            'location', 'email', 'hireable', 'bio', 'public_repos',
                            'public_gists', 'followers', 'following', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/users/{username}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def list_user_repos(username: str) -> List[Dict[str, Any]]:
        """
        List public repositories for the specified user.

        Args:
            username (str): The GitHub username of the user.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing information
                                  about a public repository. Keys may include 'id',
                                  'node_id', 'name', 'full_name', 'private', 'owner',
                                  'html_url', 'description', 'fork', 'url', 'created_at',
                                  'updated_at', 'pushed_at', 'homepage', 'size',
                                  'stargazers_count', 'watchers_count', 'language',
                                  'forks_count', 'open_issues_count', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/users/{username}/repos"
        response = requests.get(url, headers=headers)
        response.raise_for_status() 
        return response.json()

    @staticmethod
    def list_repo_issues(owner: str, repo: str, state: str = "open") -> List[Dict[str, Any]]:
        """
        List issues in the specified public repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            state (str, optional): The state of the issues to return. Can be either 'open', 'closed', or 'all'. Defaults to 'open'.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing essential information about an issue.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}/issues"
        params = {"state": state}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        def simplify_issue(issue: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "number": issue["number"],
                "title": issue["title"],
                "state": issue["state"],
                "created_at": issue["created_at"],
                "updated_at": issue["updated_at"],
                "html_url": issue["html_url"],
                "user": {
                    "login": issue["user"]["login"],
                    "id": issue["user"]["id"]
                },
                "comments": issue["comments"],
                "pull_request": "pull_request" in issue
            }

        return [simplify_issue(issue) for issue in response.json()]
    
    @staticmethod
    def get_issue_comments(owner: str, repo: str, issue_number: int) -> List[Dict[str, Any]]:
        """
        Get essential information about an issue and its comments in a repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            issue_number (int): The number of the issue.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, containing the issue description and all comments.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        
        # Get issue details
        issue_url = f"{base_url}/repos/{owner}/{repo}/issues/{issue_number}"
        issue_response = requests.get(issue_url, headers=headers)
        issue_response.raise_for_status()
        issue_data = issue_response.json()
        
        # Get comments
        comments_url = f"{issue_url}/comments"
        comments_response = requests.get(comments_url, headers=headers)
        comments_response.raise_for_status()
        comments_data = comments_response.json()
        
        def simplify_data(data: Dict[str, Any], is_issue: bool = False) -> Dict[str, Any]:
            return {
                "id": data["id"],
                "user": {
                    "login": data["user"]["login"],
                    "id": data["user"]["id"]
                },
                "created_at": data["created_at"],
                "updated_at": data["updated_at"],
                "body": data["body"],
                "type": "issue" if is_issue else "comment"
            }

        result = [simplify_data(issue_data, is_issue=True)]
        result.extend([simplify_data(comment) for comment in comments_data])
        
        return result

    @staticmethod
    def get_repo_details(owner: str, repo: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific GitHub repository.

        Args:
            owner (str): The username or organization name that owns the repository.
            repo (str): The name of the repository.

        Returns:
            Dict[str, Any]: A dictionary containing detailed information about the repository.
                            Keys may include 'id', 'node_id', 'name', 'full_name', 'private',
                            'owner', 'html_url', 'description', 'fork', 'url', 'created_at',
                            'updated_at', 'pushed_at', 'homepage', 'size', 'stargazers_count',
                            'watchers_count', 'language', 'forks_count', 'open_issues_count',
                            'master_branch', 'default_branch', 'topics', 'has_issues', 'has_projects',
                            'has_wiki', 'has_pages', 'has_downloads', 'archived', 'disabled', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def list_repo_contributors(owner: str, repo: str) -> List[Dict[str, Any]]:
        """
        List contributors to a specific GitHub repository.

        Args:
            owner (str): The username or organization name that owns the repository.
            repo (str): The name of the repository.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing information about a contributor.
                                  Keys may include 'login', 'id', 'node_id', 'avatar_url', 'gravatar_id',
                                  'url', 'html_url', 'followers_url', 'following_url', 'gists_url',
                                  'starred_url', 'subscriptions_url', 'organizations_url', 'repos_url',
                                  'events_url', 'received_events_url', 'type', 'site_admin',
                                  'contributions', etc.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}/contributors"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_repo_readme(owner: str, repo: str) -> Dict[str, str]:
        """
        Get the README content of a GitHub repository.

        Args:
            owner (str): The username or organization name that owns the repository.
            repo (str): The name of the repository.

        Returns:
            Dict[str, str]: A dictionary containing the README content.
                            The key is 'content' and the value is the raw text of the README file.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.

        Note:
            This method retrieves the raw content of the README file, regardless of its format
            (e.g., .md, .rst, .txt). The content is not rendered or processed in any way.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/repos/{owner}/{repo}/readme"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return {"content": response.text}
    
    @staticmethod
    def search_repositories(query: str, sort: str = "stars", max_results: int = 10) -> Dict[str, Any]:
        """
        Search for repositories on GitHub with a maximum number of results.
        
        Args:
            query (str): Search keywords and qualifiers.
            sort (str): Can be one of: stars, forks, help-wanted-issues, updated. Default: stars
            max_results (int): Maximum number of results to return. Default: 10

        Returns:
            Dict[str, Any]: Dictionary containing search results and metadata.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"} 
        url = f"{base_url}/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": min(max_results, 100)  # GitHub API allows max 100 items per page
        }
        
        results = []
        while len(results) < max_results:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            results.extend(data['items'][:max_results - len(results)])
            
            if 'next' not in response.links:
                break
            
            url = response.links['next']['url']
            params = {}  # Clear params as they're included in the next URL

        def simplify_repo(repo: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "id": repo["id"],
                "name": repo["name"],
                "full_name": repo["full_name"],
                "owner": {
                    "login": repo["owner"]["login"],
                    "id": repo["owner"]["id"]
                },
                "html_url": repo["html_url"],
                "description": repo["description"],
                "created_at": repo["created_at"],
                "updated_at": repo["updated_at"],
                "stargazers_count": repo["stargazers_count"],
                "forks_count": repo["forks_count"],
                "language": repo["language"],
                "topics": repo["topics"],
                "license": repo["license"]["name"] if repo["license"] else None,
                "open_issues_count": repo["open_issues_count"]
            }

        simplified_results = [simplify_repo(repo) for repo in results]

        return {
            "total_count": data['total_count'],
            "incomplete_results": data['incomplete_results'],
            "items": simplified_results[:max_results]
        }


    @staticmethod
    def get_repo_contents(owner: str, repo: str, path: str = "") -> List[Dict[str, Any]]:
        """
        Get contents of a repository directory or file.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            path (str, optional): The directory or file path. Defaults to root directory.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing information about the contents.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        url = f"{base_url}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def get_file_content(owner: str, repo: str, path: str) -> str:
        """
        Get the content of a specific file in the repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            path (str): The file path within the repository.

        Returns:
            str: The content of the file.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        url = f"{base_url}/repos/{owner}/{repo}/contents/{path}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        content = response.json()["content"]
        return base64.b64decode(content).decode('utf-8')

    @staticmethod
    def get_directory_structure(owner: str, repo: str, path: str = "") -> Dict[str, Any]:
        """
        Get the directory structure of a repository.

        Args:
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            path (str, optional): The directory path. Defaults to root directory.

        Returns:
            Dict[str, Any]: A nested dictionary representing the directory structure.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        contents = GitHubTools.get_repo_contents(owner, repo, path)
        structure = {}
        for item in contents:
            if item['type'] == 'dir':
                structure[item['name']] = GitHubTools.get_directory_structure(owner, repo, item['path'])
            else:
                structure[item['name']] = item['type']
        return structure

    @staticmethod
    def search_code(query: str, owner: str, repo: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Search for code within a specific repository.

        Args:
            query (str): The search query.
            owner (str): The owner (user or organization) of the repository.
            repo (str): The name of the repository.
            max_results (int, optional): Maximum number of results to return. Defaults to 10.

        Returns:
            Dict[str, Any]: A dictionary containing search results and metadata.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
        """
        base_url = "https://api.github.com"
        headers = {"Accept": "application/vnd.github+json"}
        url = f"{base_url}/search/code"
        params = {
            "q": f"{query} repo:{owner}/{repo}",
            "per_page": min(max_results, 100)
        }

        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()

        def simplify_code_result(item: Dict[str, Any]) -> Dict[str, Any]:
            return {
                "name": item["name"],
                "path": item["path"],
                "sha": item["sha"],
                "url": item["html_url"],
                "repository": {
                    "name": item["repository"]["name"],
                    "full_name": item["repository"]["full_name"],
                    "owner": item["repository"]["owner"]["login"]
                }
            }

        simplified_results = [simplify_code_result(item) for item in data['items'][:max_results]]

        return {
            "total_count": data['total_count'],
            "incomplete_results": data['incomplete_results'],
            "items": simplified_results
        }


class AmadeusTools:
    @staticmethod
    def _get_access_token():
        """Get Amadeus API access token."""
        load_dotenv()
        api_key = os.getenv("AMADEUS_API_KEY")
        api_secret = os.getenv("AMADEUS_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("AMADEUS_API_KEY and AMADEUS_API_SECRET must be set in .env file")
        
        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": api_secret
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()["access_token"]

    @staticmethod
    def search_flights(
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        travel_class: Optional[str] = None,
        non_stop: bool = False,
        currency: str = "USD",
        max_price: Optional[int] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for flight offers using Amadeus API.

        Args:
            origin (str): Origin airport (e.g., "YYZ")
            destination (str): Destination airport (e.g., "CDG")
            departure_date (str): Departure date in YYYY-MM-DD format
            return_date (Optional[str]): Return date in YYYY-MM-DD format for round trips
            adults (int): Number of adult travelers
            children (int): Number of child travelers
            infants (int): Number of infant travelers
            travel_class (Optional[str]): Preferred travel class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
            non_stop (bool): If True, search for non-stop flights only
            currency (str): Currency code for pricing (default: USD)
            max_price (Optional[int]): Maximum price per traveler
            max_results (int): Maximum number of results to return (default: 10)

        Returns:
            Dict[str, Any]: Flight search results
        """
        access_token = AmadeusTools._get_access_token()
        
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "children": children,
            "infants": infants,
            "currencyCode": currency,
            "max": max_results
        }
        
        if return_date:
            params["returnDate"] = return_date
        
        if travel_class:
            params["travelClass"] = travel_class
        
        if non_stop:
            params["nonStop"] = "true"
        
        if max_price:
            params["maxPrice"] = max_price
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"Error searching for flight offers: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f"\nResponse status code: {e.response.status_code}"
                error_message += f"\nResponse content: {e.response.text}"
            return error_message

    @staticmethod
    def get_cheapest_date(
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1
    ) -> Dict[str, Any]:
        """
        Find the cheapest travel dates for a given route using the Flight Offers Search API.

        Args:
            origin (str): IATA code of the origin airport.
            destination (str): IATA code of the destination airport.
            departure_date (str): Departure date in YYYY-MM-DD format.
            return_date (Optional[str]): Return date in YYYY-MM-DD format for round trips. Defaults to None.
            adults (int): Number of adult travelers. Defaults to 1.

        Returns:
            Dict[str, Any]: A dictionary containing the cheapest flight offer information, including:
                - price: The total price of the cheapest offer.
                - departureDate: The departure date of the cheapest offer.
                - returnDate: The return date of the cheapest offer (if applicable).
                - airline: The airline code of the cheapest offer.
                - additional details about the flight itinerary.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.

        Note:
            This method uses the Amadeus Flight Offers Search API to find the cheapest flight
            for the given route and dates. It returns only one result (the cheapest offer).
        """
        access_token = AmadeusTools._get_access_token()
        
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "max": 1  # We only need the cheapest offer
        }
        
        if return_date:
            params["returnDate"] = return_date
        
        response = requests.get(url, headers=headers, params=params)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response content: {response.text}")
            raise
        return response.json()

    @staticmethod
    def get_flight_inspiration(
        origin: str,
        max_price: Optional[int] = None,
        currency: str = "EUR"
    ) -> Dict[str, Any]:
        """
        Get flight inspiration using the Flight Inspiration Search API.

        This method uses the Amadeus Flight Inspiration Search API to find travel destinations
        based on the origin city and optional price constraints.

        Args:
            origin (str): IATA code of the origin city.
            max_price (Optional[int], optional): Maximum price for the flights. Defaults to None.
            currency (str, optional): Currency for the price. Defaults to "EUR".

        Returns:
            Dict[str, Any]: A dictionary containing flight inspiration results, including:
                - data: A list of dictionaries, each representing a destination with details such as:
                    - type: The type of the result (usually "flight-destination").
                    - origin: The IATA code of the origin city.
                    - destination: The IATA code of the destination city.
                    - departureDate: The suggested departure date.
                    - returnDate: The suggested return date.
                    - price: The price information for the trip.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
            ValueError: If the required environment variables are not set.

        Note:
            This method requires valid Amadeus API credentials to be set in the environment variables.
        """
        access_token = AmadeusTools._get_access_token()
        
        url = "https://test.api.amadeus.com/v1/shopping/flight-destinations"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "origin": origin,
            "currency": currency
        }
        
        if max_price:
            params["maxPrice"] = max_price
        
        response = requests.get(url, headers=headers, params=params)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response content: {response.text}")
            raise
        return response.json()


class AudioTools:
    @staticmethod
    def transcribe_audio(
        audio_input: Union[str, List[str]],
        model: str = "whisper-1",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[Literal["segment", "word"]]] = None
    ) -> Union[Dict, List[Dict]]:
        """
        Transcribe audio using the OpenAI Whisper API.

        Args:
            audio_input (Union[str, List[str]]): Path to audio file(s) or list of paths.
            model (str): The model to use for transcription. Default is "whisper-1".
            language (Optional[str]): The language of the input audio. If None, Whisper will auto-detect.
            prompt (Optional[str]): An optional text to guide the model's style or continue a previous audio segment.
            response_format (str): The format of the transcript output. Default is "json".
            temperature (float): The sampling temperature, between 0 and 1. Default is 0.
            timestamp_granularities (Optional[List[str]]): List of timestamp granularities to include.

        Returns:
            Union[Dict, List[Dict]]: Transcription result(s) in the specified format.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env file")

        url = 'https://api.openai.com/v1/audio/transcriptions'
        headers = {'Authorization': f'Bearer {api_key}'}

        def process_single_file(file_path):
            with open(file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {
                    'model': model,
                    'response_format': response_format,
                    'temperature': temperature,
                }
                if language:
                    data['language'] = language
                if prompt:
                    data['prompt'] = prompt
                if timestamp_granularities:
                    data['timestamp_granularities'] = timestamp_granularities

                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                if response_format == 'json' or response_format == 'verbose_json':
                    return response.json()
                else:
                    return response.text

        if isinstance(audio_input, str):
            return process_single_file(audio_input)
        elif isinstance(audio_input, list):
            return [process_single_file(file) for file in audio_input if os.path.isfile(file)]
        else:
            raise ValueError('Invalid input type. Expected string or list of strings.')

    @staticmethod
    def translate_audio(
        audio_input: Union[str, List[str]],
        model: str = "whisper-1",
        prompt: Optional[str] = None,
        response_format: Literal["json", "text", "srt", "verbose_json", "vtt"] = "json",
        temperature: float = 0,
        timestamp_granularities: Optional[List[Literal["segment", "word"]]] = None
    ) -> Union[Dict, List[Dict]]:
        """
        Translate audio to English using the OpenAI Whisper API.

        Args:
            audio_input (Union[str, List[str]]): Path to audio file(s) or list of paths.
            model (str): The model to use for translation. Default is "whisper-1".
            prompt (Optional[str]): An optional text to guide the model's style or continue a previous audio segment.
            response_format (str): The format of the transcript output. Default is "json".
            temperature (float): The sampling temperature, between 0 and 1. Default is 0.
            timestamp_granularities (Optional[List[str]]): List of timestamp granularities to include.

        Returns:
            Union[Dict, List[Dict]]: Translation result(s) in the specified format.
        """
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in .env file")

        url = 'https://api.openai.com/v1/audio/translations'
        headers = {'Authorization': f'Bearer {api_key}'}

        def process_single_file(file_path):
            with open(file_path, 'rb') as audio_file:
                files = {'file': audio_file}
                data = {
                    'model': model,
                    'response_format': response_format,
                    'temperature': temperature,
                }
                if prompt:
                    data['prompt'] = prompt
                if timestamp_granularities:
                    data['timestamp_granularities'] = timestamp_granularities

                response = requests.post(url, headers=headers, files=files, data=data)
                response.raise_for_status()
                
                if response_format == 'json' or response_format == 'verbose_json':
                    return response.json()
                else:
                    return response.text

        if isinstance(audio_input, str):
            return process_single_file(audio_input)
        elif isinstance(audio_input, list):
            return [process_single_file(file) for file in audio_input if os.path.isfile(file)]
        else:
            raise ValueError('Invalid input type. Expected string or list of strings.')


class CalculatorTools:
    @staticmethod
    def basic_math(operation: str, args: list) -> float:
        """
        Perform basic and advanced math operations on multiple numbers.

        Args:
            operation (str): One of 'add', 'subtract', 'multiply', 'divide', 'exponent', 'root', 'modulo', or 'factorial'.
            args (list): List of numbers to perform the operation on.

        Returns:
            float: Result of the operation.

        Raises:
            ValueError: If an invalid operation is provided, if dividing by zero, if fewer than required numbers are provided, or for invalid inputs.

        Note:
            This method does not take in letters or words. It only takes in numbers.
        """
        if len(args) < 1:
            raise ValueError("At least one number is required for the operation.")

        # Convert all args to float, except for factorial which requires int
        if operation != 'factorial':
            args = [float(arg) for arg in args]
        
        result = args[0]

        if operation in ['add', 'subtract', 'multiply', 'divide']:
            if len(args) < 2:
                raise ValueError("At least two numbers are required for this operation.")

            if operation == 'add':
                for num in args[1:]:
                    result += num
            elif operation == 'subtract':
                for num in args[1:]:
                    result -= num
            elif operation == 'multiply':
                for num in args[1:]:
                    result *= num
            elif operation == 'divide':
                for num in args[1:]:
                    if num == 0:
                        raise ValueError("Cannot divide by zero")
                    result /= num
        elif operation == 'exponent':
            if len(args) != 2:
                raise ValueError("Exponent operation requires exactly two numbers.")
            result = args[0] ** args[1]
        elif operation == 'root':
            if len(args) != 2:
                raise ValueError("Root operation requires exactly two numbers.")
            if args[1] == 0:
                raise ValueError("Cannot calculate 0th root")
            result = args[0] ** (1 / args[1])
        elif operation == 'modulo':
            if len(args) != 2:
                raise ValueError("Modulo operation requires exactly two numbers.")
            if args[1] == 0:
                raise ValueError("Cannot perform modulo with zero")
            result = args[0] % args[1]
        elif operation == 'factorial':
            if len(args) != 1 or args[0] < 0 or not isinstance(args[0], int):
                raise ValueError("Factorial operation requires exactly one non-negative integer.")
            result = 1
            for i in range(1, args[0] + 1):
                result *= i
        else:
            raise ValueError("Invalid operation. Choose 'add', 'subtract', 'multiply', 'divide', 'exponent', 'root', 'modulo', or 'factorial'.")

        # Convert the result to a string before returning
        return str(result)

    @staticmethod
    def get_current_time() -> str:
        """
        Get the current UTC time.

        Returns:
            str: The current UTC time in the format 'YYYY-MM-DD HH:MM:SS'.
        """
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def add_days(date_str: str, days: int) -> str:
        """
        Add a number of days to a given date.

        Args:
            date_str (str): The starting date in 'YYYY-MM-DD' format.
            days (int): The number of days to add (can be negative).

        Returns:
            str: The resulting date in 'YYYY-MM-DD' format.
        """
        date = datetime.strptime(date_str, "%Y-%m-%d")
        new_date = date + timedelta(days=days)
        return new_date.strftime("%Y-%m-%d")

    @staticmethod
    def days_between(date1_str: str, date2_str: str) -> int:
        """
        Calculate the number of days between two dates.

        Args:
            date1_str (str): The first date in 'YYYY-MM-DD' format.
            date2_str (str): The second date in 'YYYY-MM-DD' format.

        Returns:
            int: The number of days between the two dates.
        """
        date1 = datetime.strptime(date1_str, "%Y-%m-%d")
        date2 = datetime.strptime(date2_str, "%Y-%m-%d")
        return abs((date2 - date1).days)

    @staticmethod
    def format_date(date_str: str, input_format: str, output_format: str) -> str:
        """
        Convert a date string from one format to another.

        Args:
            date_str (str): The date string to format.
            input_format (str): The current format of the date string.
            output_format (str): The desired output format.

        Returns:
            str: The formatted date string.

        Example:
            format_date("2023-05-15", "%Y-%m-%d", "%B %d, %Y") -> "May 15, 2023"
        """
        date_obj = datetime.strptime(date_str, input_format)
        return date_obj.strftime(output_format)


class LangchainTools:
    @staticmethod
    def _wrap(langchain_tool: Type[BaseTool]) -> Callable:
        def wrapped_tool(**kwargs: Any) -> str:
            tool_instance = langchain_tool()
            # Convert kwargs to a single string input
            tool_input = json.dumps(kwargs)
            return tool_instance.run(tool_input)
        
        tool_instance = langchain_tool()
        name = getattr(tool_instance, 'name', langchain_tool.__name__)
        description = getattr(tool_instance, 'description', "No description available")
        
        doc_parts = []
        doc_parts.append(f"- {name}:")
        doc_parts.append(f"    Description: {description}")
        
        args_schema = getattr(langchain_tool, 'args_schema', None) or getattr(tool_instance, 'args_schema', None)
        if args_schema and issubclass(args_schema, BaseModel):
            doc_parts.append("    Arguments:")
            for field_name, field in args_schema.__fields__.items():
                field_desc = field.field_info.description or "No description"
                doc_parts.append(f"      - {field_name}: {field_desc}")
        
        wrapped_tool.__name__ = name
        wrapped_tool.__doc__ = "\n".join(doc_parts)
        return wrapped_tool
    
    @classmethod
    def get_tool(cls, tool_name: str) -> Callable:
        """
        Retrieve and wrap a specified Langchain tool.

        Args:
            tool_name (str): Name of the Langchain tool to retrieve.

        Returns:
            Callable: A wrapped Langchain tool as a callable function.

        Raises:
            ValueError: If an unknown tool name is provided.

        Example:
            >>> tool = LangchainTools.get_tool("WikipediaQueryRun")
        """
        if tool_name not in _module_lookup:
            raise ValueError(f"Unknown Langchain tool: {tool_name}")
        
        module_path = _module_lookup[tool_name]
        module = __import__(module_path, fromlist=[tool_name])
        tool_class = getattr(module, tool_name)
        
        wrapped_tool = LangchainTools._wrap(tool_class)
        return wrapped_tool

    @classmethod
    def list_available_tools(cls) -> List[str]:
        """
        List all available Langchain tools.

        Returns:
            List[str]: A list of names of all available Langchain tools.

        Raises:
            ImportError: If langchain-community is not installed.

        Example:
            >>> tools = LangchainTools.list_available_tools()
            >>> "WikipediaQueryRun" in tools
            True
        """
        try:
            from langchain_community.tools import _module_lookup
        except ImportError:
            print("Error: langchain-community is not installed. Please install it using 'pip install langchain-community'.")
            return []
        
        return list(_module_lookup.keys())

    @classmethod
    def get_tool_info(cls, tool_name: str) -> dict:
        """
        Retrieve information about a specific Langchain tool.

        Args:
            tool_name (str): The name of the Langchain tool.

        Returns:
            dict: A dictionary containing the tool's name, description, and module path.

        Raises:
            ValueError: If an unknown tool name is provided.

        Example:
            >>> info = LangchainTools.get_tool_info("WikipediaQueryRun")
            >>> "name" in info and "description" in info and "module_path" in info
            True
        """
        if tool_name not in _module_lookup:
            raise ValueError(f"Unknown Langchain tool: {tool_name}")
        
        module_path = _module_lookup[tool_name]
        module = __import__(module_path, fromlist=[tool_name])
        tool_class = getattr(module, tool_name)
        
        tool_instance = tool_class()
        name = getattr(tool_instance, 'name', tool_class.__name__)
        description = getattr(tool_instance, 'description', "No description available")
        
        return {
            "name": name,
            "description": description,
            "module_path": module_path
        }
