import os, time, requests, base64, random
from typing import List, Union, Tuple, Optional
from anthropic import Anthropic, APIStatusError, APITimeoutError, APIConnectionError, APIResponseValidationError, RateLimitError
from openai.types.chat import ChatCompletion
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError
from groq import Groq
from groq.types.chat import ChatCompletion
import ollama
from halo import Halo
import threading
import json
from dotenv import load_dotenv
load_dotenv()

# Global settings
verbosity = False
debug = False

def set_verbosity(value: Union[str, bool, int]):
    global verbosity, debug
    if isinstance(value, str):
        value = value.lower()
        if value in ['debug', '2']:
            verbosity = True
            debug = True
        elif value in ['true', '1']:
            verbosity = True
            debug = False
        else:
            verbosity = False
            debug = False
    elif isinstance(value, bool):
        verbosity = value
        debug = False
    elif isinstance(value, int):
        if value == 2:
            verbosity = True
            debug = True
        elif value == 1:
            verbosity = True
            debug = False
        else:
            verbosity = False
            debug = False


# Define color codes
COLORS = {
    'cyan': '\033[96m',
    'blue': '\033[94m',
    'green': '\033[92m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'reset': '\033[0m'
}

def print_color(message, color):
    print(f"{COLORS.get(color, '')}{message}{COLORS['reset']}")

def print_api_request(message):
    if verbosity:
        print_color(message, 'green')

def print_model_request(provider: str, model: str):
    if verbosity:
        print_color(f"Sending request to {model} from {provider}", 'cyan')

def print_label(message:str):
    if verbosity:
        print_color(message, 'cyan')

def print_api_response(message):
    if verbosity:
        print_color(message, 'blue')

def print_debug(message):
    if debug:
        print_color(message, 'yellow')

def print_error(message):
    print_color(message, 'red')


class OpenaiModels:
    @staticmethod
    def send_openai_request(system_prompt: str, user_prompt: str, model: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        print_model_request("OpenAI", model)
        if debug:
            print_debug(f"Entering send_openai_request function")
            print_debug(f"Parameters: system_prompt={system_prompt}, user_prompt={user_prompt}, model={model}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print_debug(f"OpenAI client initialized with API key: {os.getenv('OPENAI_API_KEY')[:5]}...")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        print_debug(f"Initial messages: {messages}")

        if image_data:
            print_debug("Processing image data")
            if isinstance(image_data, str):
                image_data = [image_data]
            
            for i, image in enumerate(image_data, start=1):
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": f"Image {i}:"}, {"type": "image_url", "image_url": {"url": image}}]
                })
            print_debug(f"Messages after adding image data: {messages}")

        max_retries = 6
        base_delay = 5
        max_delay = 60

        print_api_request(f"{system_prompt}\n{user_prompt}")
        if image_data:
            print_api_request("Images: Included")

        spinner = Halo(text='Sending request to OpenAI...', spinner='dots')
        stop_spinner = threading.Event()

        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        try:
            for attempt in range(max_retries):
                print_debug(f"Attempt {attempt + 1}/{max_retries}")
                try:
                    print_debug("Creating chat completion")
                    completion_params = {
                        "messages": messages,
                        "model": model,
                        "max_tokens": max_tokens,
                        "temperature": temperature
                    }
                    if require_json_output:
                        completion_params["response_format"] = {"type": "json_object"}
                        if not any('json' in msg['content'].lower() for msg in messages if isinstance(msg.get('content'), str)):
                            messages.append({"role": "user", "content": "Please provide your response in valid JSON format."})
                    
                    response: ChatCompletion = client.chat.completions.create(**completion_params)
                    print_debug(f"Response received: {response}")

                    response_text = response.choices[0].message.content if response.choices else ""
                    
                    if require_json_output:
                        try:
                            json_response = json.loads(response_text)
                            return json.dumps(json_response), None
                        except json.JSONDecodeError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                    
                    return response_text.strip(), None

                except RateLimitError as e:
                    print_error(f"Rate limit exceeded: {e}")
                    print_debug(f"RateLimitError details: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except AuthenticationError as e:
                    print_error(f"Authentication error: {e}")
                    print_debug(f"AuthenticationError details: {e}")
                    return "", e

                except (APIConnectionError, APITimeoutError) as e:
                    print_error(f"API Connection/Timeout error: {e}")
                    print_debug(f"{type(e).__name__} details: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except APIError as e:
                    print_error(f"API error: {e}")
                    print_debug(f"APIError details: {e}")
                    if e.status_code in [400, 404]:  # Bad request or Not Found
                        return "", e  # Don't retry these errors
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except Exception as e:
                    print_error(f"An unexpected error occurred: {e}")
                    print_debug(f"Unexpected error details: {type(e).__name__}, {e}")
                    return "", e  # Don't retry unexpected errors

            print_debug("Exiting send_openai_request function with empty string")
            return "", Exception("Max retries reached")

        finally:
            stop_spinner.set()
            spinner_thread.join()
            if 'response_text' in locals() and response_text:
                spinner.succeed('Request completed')
                print_api_response(response_text.strip())
            else:
                spinner.fail('Request failed')

    @staticmethod
    def gpt_4_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4-turbo", image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def gpt_3_5_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-3.5-turbo", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gpt_4o(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4o", image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def gpt_4o_mini(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4o-mini", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gpt_4(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4", image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def custom_model(model_name: str):
        def wrapper(system_prompt: str = "", user_prompt: str = "", image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
            return OpenaiModels.send_openai_request(system_prompt, user_prompt, model_name, image_data, temperature, max_tokens, require_json_output)
        return wrapper

class AnthropicModels:
    @staticmethod
    def call_anthropic(system_prompt: str, user_prompt: str, model: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        print_model_request("Anthropic", model)
        if debug:
            print_debug(f"Entering call_anthropic function")
            print_debug(f"Parameters: system_prompt={system_prompt}, user_prompt={user_prompt}, model={model}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        max_retries = 6
        base_delay = 5
        max_delay = 60

        print_api_request(f"{system_prompt}\n{user_prompt}")
        if image_data:
            print_api_request("Images: (base64-encoded)")

        spinner = Halo(text='Sending request to Anthropic...', spinner='dots')
        stop_spinner = threading.Event()

        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        try:
            for attempt in range(max_retries):
                try:
                    print_debug(f"Attempt {attempt + 1} of {max_retries}")
                    
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if not api_key:
                        return "", ValueError("ANTHROPIC_API_KEY environment variable is not set")
                    
                    print_debug(f"API Key: {api_key[:5]}...{api_key[-5:]}")
                    print_debug(f"Anthropic client initialized")
                    print_debug(f"Model: {model}")

                    client = Anthropic(api_key=api_key)
                    messages = [{"role": "user", "content": []}]
                    
                    if image_data:
                        print_debug(f"Processing image data")
                        if isinstance(image_data, str):
                            image_data = [image_data]
                        
                        for i, image in enumerate(image_data, start=1):
                            try:
                                image_media_type = "image/png"  # Default to PNG
                            except:
                                image_media_type = "image/png"  # Default to PNG if detection fails
                            
                            print_debug(f"Image {i} media type: {image_media_type}")
                            
                            messages[0]["content"].append({"type": "text", "text": f"Image {i}:"})
                            messages[0]["content"].append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": image_media_type,
                                    "data": image,  # Use the full base64 string, not truncated
                                },
                            })
                        
                        messages[0]["content"].append({"type": "text", "text": user_prompt})
                    else:
                        messages[0]["content"] = user_prompt

                    if require_json_output:
                        if not any('json' in msg['content'].lower() for msg in messages if isinstance(msg.get('content'), str)):
                            messages.append({"role": "user", "content": "Please provide your response in valid JSON format."})
                    
                    print_debug(f"Final messages structure: {messages}")
                    
                    message = client.messages.create(
                        model=model,
                        system=system_prompt,
                        max_tokens=max_tokens,
                        temperature=temperature, 
                        messages=messages
                    )
                    print_debug(f"API response received")
                    
                    response_texts = [block.text for block in message.content if block.type == 'text']
                    response_text = " ".join(response_texts)
                    print_debug(f"Processed response text (truncated): {response_text[:100]}...")
                    
                    if require_json_output:
                        try:
                            json.loads(response_text)  # Attempt to parse JSON, but don't store the result
                        except json.JSONDecodeError:
                            print_color("Warning: Response is not valid JSON. Returning raw text.", 'yellow')
                    
                    return response_text, None

                except APIStatusError as e:
                    print_debug(f"API status error: {e}")
                    if e.status_code in [400, 401, 404]:
                        error_messages = {
                            400: "Bad request. Please check your input parameters.",
                            401: "Authentication error. Please check your API key.",
                            404: "Not found error. The specified model may not exist."
                        }
                        print_error(f"{error_messages[e.status_code]} Error details: {e}")
                        return "", e  # Don't retry for these errors
                    elif e.status_code == 429:
                        print_error(f"Rate limit exceeded: {e}")
                        # Continue to retry logic
                    else:
                        print_error(f"Unexpected API error: {e}")
                        # Continue to retry logic

                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_error(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except (APITimeoutError, APIConnectionError) as e:
                    print_debug(f"{type(e).__name__}: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_error(f"{type(e).__name__} occurred. Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except RateLimitError as e:
                    print_error(f"Rate limit exceeded: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_error(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except Exception as e:
                    print_debug(f"Unexpected error: {e}")
                    print_error(f"An unexpected error occurred: {e}")
                    return "", e  # Don't retry for unexpected errors

            print_debug("Max retries reached")
            print_error("Max retries reached. Unable to get a response from the Anthropic API.")
            return "", Exception("Max retries reached")

        finally:
            stop_spinner.set()
            spinner_thread.join()
            if 'response_text' in locals() and response_text:
                spinner.succeed('Request completed')
                print_api_response(response_text)
            else:
                spinner.fail('Request failed')

    @staticmethod
    def opus(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-opus-20240229", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def sonnet(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-sonnet-20240229", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def haiku(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-haiku-20240307", image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def sonnet_3_5(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-5-sonnet-20240620", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def custom_model(model_name: str):
        def wrapper(system_prompt: str = "", user_prompt: str = "", image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
            return AnthropicModels.call_anthropic(system_prompt, user_prompt, model_name, image_data, temperature, max_tokens, require_json_output)
        return wrapper

class OpenrouterModels:
    @staticmethod
    def call_openrouter_api(model_key: str, system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        print_model_request("OpenRouter", model_key)
        if debug:
            print_debug(f"Entering call_openrouter_api function")
            print_debug(f"Parameters: model_key={model_key}, system_prompt={system_prompt}, user_prompt={user_prompt}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        print_api_request(f"{system_prompt}\n{user_prompt}")
        if image_data:
            print_api_request(f"Image data included: {len(image_data) if isinstance(image_data, list) else 1} image(s)")

        spinner = Halo(text='Sending request to OpenRouter...', spinner='dots')
        stop_spinner = threading.Event()

        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        max_retries = 6
        base_delay = 5
        max_delay = 60

        for attempt in range(max_retries):
            try:
                # Load environment variables from .env file
                load_dotenv()
                print_debug("Environment variables loaded")

                api_key = os.getenv("OPENROUTER_API_KEY")
                print_debug(f"API key retrieved: {'*' * len(api_key) if api_key else 'None'}")
                if not api_key:
                    return "", ValueError("Openrouter API key not found in environment variables")

                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": os.getenv("taskflowai.org", ""),  # Optional, for including your app on openrouter.ai rankings
                    "X-Title": os.getenv("taskflowai.org", "")  # Optional, shows in rankings on openrouter.ai
                }
                print_debug(f"Headers prepared: {headers}")

                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": []}
                ]
                print_debug(f"Initial messages structure: {messages}")

                if image_data:
                    print_debug("Processing image data")
                    if isinstance(image_data, str):
                        image_data = [image_data]
                    
                    for i, image in enumerate(image_data, start=1):
                        messages[1]["content"].append({"type": "text", "text": f"Image {i}:"})
                        if image.startswith(('http://', 'https://')):
                            print_debug(f"Image {i} is a URL")
                            messages[1]["content"].append({
                                "type": "image_url",
                                "image_url": {"url": image}
                            })
                        else:
                            print_debug(f"Image {i} is base64, attempting to detect image type")
                            try:
                                image_data = base64.b64decode(image)
                                image_type = None
                                if image_data.startswith(b'\xff\xd8'):
                                    image_type = 'jpeg'
                                elif image_data.startswith(b'\x89PNG\r\n\x1a\n'):
                                    image_type = 'png'
                                elif image_data.startswith(b'GIF87a') or image_data.startswith(b'GIF89a'):
                                    image_type = 'gif'
                                elif image_data.startswith(b'RIFF') and image_data[8:12] == b'WEBP':
                                    image_type = 'webp'
                                image_type = image_type if image_type else 'png'  # Default to png if detection fails
                                print_debug(f"Detected image type: {image_type}")
                            except:
                                image_type = 'png'  # Default to png if any error occurs
                                print_debug("Error detecting image type, defaulting to png")
                            
                            messages[1]["content"].append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/{image_type};base64,{image}"}
                            })

                    messages[1]["content"].append({"type": "text", "text": user_prompt})
                else:
                    messages[1]["content"] = user_prompt

                if require_json_output:
                    if not any('json' in msg['content'].lower() for msg in messages if isinstance(msg.get('content'), str)):
                        messages.append({"role": "user", "content": "Please provide your response in valid JSON format."})

                print_debug(f"Final messages structure: {messages}")
                
                body = {
                    "model": model_key,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                }

                if require_json_output:
                    body["response_format"] = {"type": "json_object"}
                    # Add an explicit instruction to output JSON
                    body["messages"].append({
                        "role": "user",
                        "content": "Please provide your response in valid JSON format."
                    })

                print_debug(f"Request body prepared: {body}")

                print_debug("Sending POST request to Openrouter API")
                response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
                print_debug(f"Response status code: {response.status_code}")
                response.raise_for_status()  # Raises an HTTPError for bad responses
                response_data = response.json()
                print_debug(f"Response data: {response_data}")
                
                if 'choices' in response_data and len(response_data['choices']) > 0:
                    generated_text = response_data['choices'][0]['message']['content']
                    print_debug(f"Generated text: {generated_text.strip()}")
                    
                    if require_json_output:
                        try:
                            json_response = json.loads(generated_text)
                            return json.dumps(json_response), None
                        except json.JSONDecodeError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                    
                    return generated_text.strip(), None
                else:
                    error_msg = f"Unexpected response format. 'choices' key not found or empty. Response data: {response_data}"
                    print_error(error_msg)
                    return "", ValueError(error_msg)
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                error_msg = f"HTTP error occurred. Status code: {status_code}. "
                if status_code in [400, 401, 402, 403, 404]:
                    error_messages = {
                        400: "Bad Request: Invalid or missing parameters. Check your input and try again.",
                        401: "Unauthorized: Invalid API key or expired OAuth session. Check your credentials.",
                        402: "Payment Required: Insufficient credits. Please add more credits to your account.",
                        403: "Forbidden: Your input was flagged by content moderation.",
                        404: "Not Found: The requested resource could not be found."
                    }
                    error_msg += error_messages.get(status_code, "Unexpected client error.")
                    print_error(error_msg)
                    return "", Exception(error_msg)  # Don't retry these errors
                elif status_code in [408, 429, 502, 503]:
                    error_messages = {
                        408: "Request Timeout: Your request took too long to process. Try again or simplify your input.",
                        429: "Too Many Requests: You are being rate limited. Please slow down your requests.",
                        502: "Bad Gateway: The chosen model is currently unavailable. Try again later or use a different model.",
                        503: "Service Unavailable: No available model provider meets your routing requirements."
                    }
                    error_msg += error_messages.get(status_code, "Unexpected server error.")
                    print_error(error_msg)
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", Exception(error_msg)
                else:
                    error_msg += f"HTTP error occurred: {e}"
                    print_error(error_msg)
                    print_error(f"Response content: {e.response.content}")
                    print_debug(f"Full exception details: {e}")
                    return "", Exception(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error occurred: {e}. Response content: {e.response.content if e.response else 'No response content'}"
                print_error(error_msg)
                print_debug(f"Full exception details: {e}")
                if attempt < max_retries - 1:
                    retry_delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    total_delay = retry_delay + jitter
                    print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)
                else:
                    return "", Exception(error_msg)
            except KeyError as e:
                error_msg = f"Key error occurred: {e}. Response data: {response_data}"
                print_error(error_msg)
                print_debug(f"Full exception details: {e}")
                return "", Exception(error_msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred: {e}"
                print_error(error_msg)
                print_debug(f"Full exception details: {e}")
                return "", Exception(error_msg)

            finally:
                stop_spinner.set()
                spinner_thread.join()
                if 'generated_text' in locals() and generated_text:
                    spinner.succeed('Request completed')
                    print_api_response(generated_text.strip())
                else:
                    spinner.fail('Request failed')

            return "", Exception("Max retries reached")

    @staticmethod
    def haiku(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3-haiku", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def sonnet(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3-sonnet", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def sonnet_3_5(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3.5-sonnet", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def opus(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3-opus", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gpt_3_5_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("openai/gpt-3.5-turbo", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gpt_4_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4-turbo", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gpt_4(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gpt_4o(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4o", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def gpt_4o_mini(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4o-mini", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def gemini_flash_1_5(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("google/gemini-flash-1.5", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_70b_sonar_32k(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("perplexity/llama-3-sonar-large-32k-chat", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def command_r(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("cohere/command-r-plus", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def nous_hermes_2_mistral_7b_dpo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("nousresearch/nous-hermes-2-mistral-7b-dpo", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def nous_hermes_2_mixtral_8x7b_dpo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("nousresearch/nous-hermes-2-mixtral-8x7b-dpo", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def nous_hermes_yi_34b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("nousresearch/nous-hermes-yi-34b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def qwen_2_72b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("qwen/qwen-2-72b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def mistral_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("mistralai/mistral-7b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def mistral_7b_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("mistralai/mistral-7b-instruct:nitro", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def mixtral_8x7b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("mistralai/mixtral-8x7b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def mixtral_8x7b_instruct_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("mistralai/mixtral-8x7b-instruct:nitro", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def mixtral_8x22b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("mistralai/mixtral-8x22b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def wizardlm_2_8x22b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("microsoft/wizardlm-2-8x22b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def neural_chat_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("intel/neural-chat-7b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gemma_7b_it(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("google/gemma-7b-it", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gemini_pro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("google/gemini-pro", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_8b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-8b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def llama_3_70b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-70b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
        
    @staticmethod
    def llama_3_70b_instruct_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-70b-instruct:nitro", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def llama_3_8b_instruct_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-8b-instruct:nitro", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def dbrx_132b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("databricks/dbrx-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def deepseek_coder(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("deepseek/deepseek-coder", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_1_70b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3.1-70b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_1_8b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3.1-8b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_1_405b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3.1-405b-instruct", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def custom_model(model_name: str):
        def wrapper(system_prompt: str = "", user_prompt: str = "", image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
            return OpenrouterModels.call_openrouter_api(model_name, system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
        return wrapper

class OllamaModels:
    @staticmethod
    def call_ollama(model: str, system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        print_model_request("Ollama", model)
        if debug:
            print_debug(f"Entering call_ollama function")
            print_debug(f"Parameters: model={model}, system_prompt={system_prompt}, user_prompt={user_prompt}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        max_retries = 6
        base_delay = 5
        max_delay = 60

        print_api_request(f"{system_prompt}\n{user_prompt}")
        if image_data:
            print_api_request("Images: Included")

        spinner = Halo(text='Sending request to Ollama...', spinner='dots')
        stop_spinner = threading.Event()

        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            if require_json_output:
                messages.append({"role": "user", "content": "Please provide your response in valid JSON format."})

            if image_data:
                print_debug("Processing image data")
                if isinstance(image_data, str):
                    image_data = [image_data]
                
                for i, image in enumerate(image_data, start=1):
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Image {i}:"},
                            {"type": "image", "image_url": {"url": image}}
                        ]
                    })

            print_debug(f"Final messages structure: {messages}")

            for attempt in range(max_retries):
                print_debug(f"Attempt {attempt + 1}/{max_retries}")
                try:
                    client = ollama.Client()
                    response = client.chat(
                        model=model,
                        messages=messages,
                        format="json" if require_json_output else None,
                        options={
                            "temperature": temperature,
                            "num_predict": max_tokens
                        }
                    )

                    response_text = response['message']['content']
                    
                    if require_json_output:
                        try:
                            json_response = json.loads(response_text)
                            return json.dumps(json_response), None
                        except json.JSONDecodeError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                    
                    return response_text.strip(), None

                except ollama.ResponseError as e:
                    print_error(f"Ollama response error: {e}")
                    print_debug(f"ResponseError details: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except ollama.RequestError as e:
                    print_error(f"Ollama request error: {e}")
                    print_debug(f"RequestError details: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except Exception as e:
                    print_error(f"An unexpected error occurred: {e}")
                    print_debug(f"Unexpected error details: {type(e).__name__}, {e}")
                    return "", e

        finally:
            stop_spinner.set()
            spinner_thread.join()
            if 'response_text' in locals() and response_text:
                spinner.succeed('Request completed')
                print_api_response(response_text.strip())
            else:
                spinner.fail('Request failed')

        return "", Exception("Max retries reached")

    # Llama 3 models
    @staticmethod
    def llama3_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("llama3", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("llama3:70b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Gemma models
    @staticmethod
    def gemma_2b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("gemma:2b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gemma_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("gemma:7b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Mistral model
    @staticmethod
    def mistral_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("mistral", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Qwen models
    @staticmethod
    def qwen_0_5b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen:0.5b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen_1_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen:1.8b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen_4b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen:4b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen_32b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen:32b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen_72b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen:72b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen_110b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen:110b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Phi-3 models
    @staticmethod
    def phi3_3b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("phi3:3b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def phi3_14b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("phi3:14b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Llama 2 models
    @staticmethod
    def llama2_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("llama2:7b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama2_13b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("llama2:13b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama2_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("llama2:70b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # CodeLlama models
    @staticmethod
    def codellama_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("codellama:7b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def codellama_13b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("codellama:13b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def codellama_34b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("codellama:34b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def codellama_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("codellama:70b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Gemma 2 models
    @staticmethod
    def gemma2_9b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("gemma2:9b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gemma2_27b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("gemma2:27b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Qwen 2 models
    @staticmethod
    def qwen2_0_5b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen2:0.5b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen2_1_5b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen2:1.5b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen2_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen2:7b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def qwen2_72b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("qwen2:72b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # LLaVA model
    @staticmethod
    def llava(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("llava", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Mixtral models
    @staticmethod
    def mixtral_8x7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("mixtral:8x7b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def mixtral_8x22b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("mixtral:8x22b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    # Dolphin Mixtral models
    @staticmethod
    def dolphin_mixtral_8x7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("dolphin-mixtral:8x7b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def dolphin_mixtral_8x22b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return OllamaModels.call_ollama("dolphin-mixtral:8x22b", system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
    
    @staticmethod
    def custom_model(model_name: str):
        def wrapper(system_prompt: str = "", user_prompt: str = "", image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
            return OllamaModels.call_ollama(model_name, system_prompt, user_prompt, image_data, temperature, max_tokens, require_json_output)
        return wrapper

class GroqModels:
    @staticmethod
    def call_groq(system_prompt: str, user_prompt: str, model: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        print_model_request("Groq", model)
        if debug:
            print_debug(f"Entering call_groq function")
            print_debug(f"Parameters: system_prompt={system_prompt}, user_prompt={user_prompt}, model={model}, image_data={image_data}, temperature={temperature}, max_tokens={max_tokens}, require_json_output={require_json_output}")

        max_retries = 6
        base_delay = 5
        max_delay = 60

        print_api_request(f"{system_prompt}\n{user_prompt}")
        if image_data:
            print_api_request("Images: Included")

        spinner = Halo(text='Sending request to Groq...', spinner='dots')
        stop_spinner = threading.Event()

        def spin():
            spinner.start()
            while not stop_spinner.is_set():
                time.sleep(0.1)
            spinner.stop()

        spinner_thread = threading.Thread(target=spin)
        spinner_thread.start()

        try:
            for attempt in range(max_retries):
                print_debug(f"Attempt {attempt + 1}/{max_retries}")
                try:
                    api_key = os.getenv("GROQ_API_KEY")
                    if not api_key:
                        return "", ValueError("GROQ_API_KEY environment variable is not set")

                    print_debug(f"API Key: {api_key[:5]}...{api_key[-5:]}")
                    client = Groq(api_key=api_key)
                    print_debug(f"Groq client initialized")

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    if require_json_output:
                        if not any('json' in msg['content'].lower() for msg in messages if isinstance(msg.get('content'), str)):
                            messages.append({"role": "user", "content": "Please provide your response in valid JSON format."})

                    if image_data:
                        print_debug("Processing image data")
                        if isinstance(image_data, str):
                            image_data = [image_data]
                        
                        for i, image in enumerate(image_data, start=1):
                            messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": f"Image {i}:"},
                                    {"type": "image_url", "image_url": {"url": image}}
                                ]
                            })

                    print_debug(f"Final messages structure: {messages}")

                    response: ChatCompletion = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format={"type": "json_object"} if require_json_output else None
                    )

                    print_debug(f"API response received")
                    
                    response_text = response.choices[0].message.content
                    print_debug(f"Processed response text (truncated): {response_text[:100]}...")
                    
                    if require_json_output:
                        try:
                            json_response = json.loads(response_text)
                            return json.dumps(json_response), None
                        except json.JSONDecodeError as e:
                            return "", ValueError(f"Failed to parse response as JSON: {e}")
                    
                    return response_text.strip(), None

                except RateLimitError as e:
                    print_error(f"Rate limit exceeded: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        jitter = random.uniform(0, 0.1 * retry_delay)
                        total_delay = retry_delay + jitter
                        print_error(f"Retrying in {total_delay:.2f} seconds...")
                        time.sleep(total_delay)
                    else:
                        return "", e

                except APITimeoutError as e:
                    print_error(f"API request timed out: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        print_error(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        return "", e

                except APIConnectionError as e:
                    print_error(f"API connection error: {e}")
                    if attempt < max_retries - 1:
                        retry_delay = min(max_delay, base_delay * (2 ** attempt))
                        print_error(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        return "", e

                except APIError as e:
                    print_error(f"API error: {e}")
                    return "", e

                except Exception as e:
                    print_error(f"An unexpected error occurred: {e}")
                    print_debug(f"Error details: {type(e).__name__}, {e}")
                    return "", e

            print_debug("Max retries reached")
            return "", Exception("Max retries reached")

        finally:
            stop_spinner.set()
            spinner_thread.join()
            if 'response_text' in locals() and response_text:
                spinner.succeed('Request completed')
                print_api_response(response_text.strip())
            else:
                spinner.fail('Request failed')


    @staticmethod
    def gemma2_9b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "gemma2-9b-it", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def gemma_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "gemma-7b-it", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_groq_70b_tool_use(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-groq-70b-8192-tool-use-preview", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_groq_8b_tool_use(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-groq-8b-8192-tool-use-preview", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_1_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 8000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama-3.1-70b-versatile", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_3_1_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 8000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama-3.1-8b-instant", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama_guard_3_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama-guard-3-8b", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llava_1_5_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llava-v1.5-7b-4096-preview", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-70b-8192", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def llama3_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "llama3-8b-8192", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def mixtral_8x7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
        return GroqModels.call_groq(system_prompt, user_prompt, "mixtral-8x7b-32768", image_data, temperature, max_tokens, require_json_output)

    @staticmethod
    def custom_model(model_name: str):
        def wrapper(system_prompt: str = "", user_prompt: str = "", image_data: Union[List[str], str, None] = None, temperature: float = 0.7, max_tokens: int = 4000, require_json_output: bool = False) -> Tuple[str, Optional[Exception]]:
            return GroqModels.call_groq(system_prompt, user_prompt, model_name, image_data, temperature, max_tokens, require_json_output)
        return wrapper
