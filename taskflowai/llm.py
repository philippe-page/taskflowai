import os, time, requests, base64, random
from typing import List, Union
from anthropic import Anthropic, APIStatusError, APITimeoutError, APIConnectionError, APIResponseValidationError, RateLimitError
from openai.types.chat import ChatCompletion
from openai import OpenAI, APIError, APIConnectionError, APITimeoutError, RateLimitError, AuthenticationError
import ollama
from dotenv import load_dotenv
load_dotenv()

# Display mode enables printing of API requests and responses
display_requests = False 
display_responses = False

# Debug mode enables comprehensive logging for detailed diagnostics
debug_mode = False

# Define color codes
COLORS = {
    'cyan': '\033[96m',
    'blue': '\033[94m',
    'yellow': '\033[93m',
    'red': '\033[91m',
    'reset': '\033[0m'
}

def print_color(message, color):
    print(f"{COLORS.get(color, '')}{message}{COLORS['reset']}")

def print_api_request(message):
    if not display_requests:
        return
    print_color(message, 'cyan')

def print_api_response(message):
    if not display_responses:
        return
    print_color(message, 'blue')

def print_debug(message):
    if not debug_mode:
        return
    print_color(message, 'yellow')

def print_error(message):
    print_color(message, 'red')


class OpenaiModels:
    @staticmethod
    def send_openai_request(system_prompt: str, user_prompt: str, model: str, image_data: Union[List[str], str, None] = None) -> str:
        print_debug(f"Entering send_openai_request function")
        print_debug(f"Parameters: system_prompt={system_prompt}, user_prompt={user_prompt}, model={model}, image_data={image_data}")

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

        for attempt in range(max_retries):
            print_debug(f"Attempt {attempt + 1}/{max_retries}")
            try:
                print_api_request(f"Sending request to {model}")
                print_api_request(f"Attempt: {attempt + 1}/{max_retries}")
                print_api_request(f"System prompt: '{system_prompt}'")
                print_api_request(f"User prompt: '{user_prompt}'")
                if image_data:
                    print_api_request("Images: Included")
                else:
                    print_api_request("Images: None")

                print_debug("Creating chat completion")
                response: ChatCompletion = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_tokens=4000,
                    temperature=0.7
                )
                print_debug(f"Response received: {response}")

                response_text = response.choices[0].message.content if response.choices else ""
                print_api_response(response_text.strip())
                print_debug(f"Returning response: {response_text.strip()}")
                return response_text.strip()

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
                    raise

            except AuthenticationError as e:
                print_error(f"Authentication error: {e}")
                print_error("Please check your OpenAI API key and ensure it's correctly set in the environment variables.")
                print_debug(f"AuthenticationError details: {e}")
                raise

            except APIConnectionError as e:
                print_error(f"API Connection error: {e}")
                print_debug(f"APIConnectionError details: {e}")
                if attempt < max_retries - 1:
                    retry_delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    total_delay = retry_delay + jitter
                    print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)
                else:
                    raise

            except APITimeoutError as e:
                print_error(f"API Timeout error: {e}")
                print_debug(f"APITimeoutError details: {e}")
                if attempt < max_retries - 1:
                    retry_delay = min(max_delay, base_delay * (2 ** attempt))
                    jitter = random.uniform(0, 0.1 * retry_delay)
                    total_delay = retry_delay + jitter
                    print_api_request(f"Retrying in {total_delay:.2f} seconds...")
                    time.sleep(total_delay)
                else:
                    raise

            except APIError as e:
                print_error(f"API error: {e}")
                print_debug(f"APIError details: {e}")
                raise

            except Exception as e:
                print_error(f"An unexpected error occurred: {e}")
                print_debug(f"Unexpected error details: {type(e).__name__}, {e}")
                raise

        print_debug("Exiting send_openai_request function with empty string")
        return ""

    @staticmethod
    def gpt_4_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4-turbo", image_data)
    
    @staticmethod
    def gpt_3_5_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-3.5-turbo", image_data)

    @staticmethod
    def gpt_4o(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4o", image_data)
    
    @staticmethod
    def gpt_4o_mini(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4o-mini", image_data)

    @staticmethod
    def gpt_4(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, "gpt-4", image_data)
    
    @staticmethod
    def custom_model(model: str, system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenaiModels.send_openai_request(system_prompt, user_prompt, model, image_data)


class AnthropicModels:
    @staticmethod
    def call_anthropic(system_prompt: str, user_prompt: str, model: str, image_data: Union[List[str], str, None] = None) -> str:
        max_retries = 6
        base_delay = 5

        for attempt in range(max_retries):
            try:
                print_debug(f"Attempt {attempt + 1} of {max_retries}")
                
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
                
                print_debug(f"API Key: {api_key[:5]}...{api_key[-5:]}")

                client = Anthropic(api_key=api_key)
                print_debug(f"Anthropic client initialized")

                print_api_request(f"Sending request to {model}")
                print_debug(f"Model: {model}")
                
                messages = [{"role": "user", "content": []}]
                
                if image_data:
                    print_debug(f"Processing image data")
                    if isinstance(image_data, str):
                        image_data = [image_data]
                    
                    for i, image in enumerate(image_data, start=1):
                        try:
                            # Remove this line as it's causing issues:
                            # image_bytes = base64.b64decode(image)
                            # Instead, assume the image is already in base64 format:
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
                    print_api_request("Prompt details:")
                    print_api_request(f"  System: {system_prompt}")
                    print_api_request(f"  User: {user_prompt}")
                    print_api_request("  Images: (base64-encoded)")
                else:
                    messages[0]["content"] = user_prompt
                    print_api_request("Prompt details:")
                    print_api_request(f"  System: {system_prompt}")
                    print_api_request(f"  User: {user_prompt}")
                
                print_debug(f"Final messages structure: {messages}")
                
                message = client.messages.create(
                    model=model,
                    system=system_prompt,
                    max_tokens=4000,
                    temperature=0.7, 
                    messages=messages
                )
                print_debug(f"API response received")
                
                response_texts = [block.text for block in message.content if block.type == 'text']
                response_text = " ".join(response_texts)
                print_debug(f"Processed response text (truncated): {response_text[:100]}...")
                
                print_api_response(response_text)
                return response_text

            except RateLimitError as e:
                print_debug(f"Rate limit error: {e}")
                retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                print_error(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except APITimeoutError:
                print_debug("API timeout error")
                retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                print_error(f"Request timed out. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except APIConnectionError:
                print_debug("API connection error")
                retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                print_error(f"Connection error. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except APIStatusError as e:
                print_debug(f"API status error: {e}")
                if e.status_code == 400:
                    print_error(f"Bad request: {e}. Please check your input parameters.")
                elif e.status_code == 401:
                    print_error("Authentication error. Please check your API key.")
                elif e.status_code == 429:
                    retry_delay = base_delay * (2 ** attempt)  # Exponential backoff
                    print_error(f"Too many requests. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print_error(f"API error: {e}")
                if attempt == max_retries - 1:
                    raise
            except APIResponseValidationError as e:
                print_debug(f"API response validation error: {e}")
                print_error(f"Invalid API response: {e}")
                if attempt == max_retries - 1:
                    raise
            except ValueError as e:
                print_debug(f"Value error: {e}")
                print_error(f"Value error: {e}")
                raise
            except Exception as e:
                print_debug(f"Unexpected error: {e}")
                print_error(f"An unexpected error occurred: {e}")
                if attempt == max_retries - 1:
                    raise

        print_debug("Max retries reached")
        print_error("Max retries reached. Unable to get a response from the Anthropic API.")
        return ""
    
    @staticmethod
    def opus(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-opus-20240229", image_data)

    @staticmethod
    def sonnet(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-sonnet-20240229", image_data)

    @staticmethod
    def haiku(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-haiku-20240307", image_data)
    
    @staticmethod
    def sonnet_3_5(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, "claude-3-5-sonnet-20240620", image_data)

    @staticmethod
    def custom_model(model: str, system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return AnthropicModels.call_anthropic(system_prompt, user_prompt, model, image_data)
    

class OpenrouterModels:
    @staticmethod
    def call_openrouter_api(model_key: str, system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        print_debug(f"Entering call_openrouter_api function")
        print_debug(f"Parameters: model_key={model_key}, system_prompt={system_prompt}, user_prompt={user_prompt}, image_data={image_data}")

        # Load environment variables from .env file
        load_dotenv()
        print_debug("Environment variables loaded")

        api_key = os.getenv("OPENROUTER_API_KEY")
        print_debug(f"API key retrieved: {'*' * len(api_key) if api_key else 'None'}")
        if not api_key:
            print_error("Openrouter API key not found in environment variables")
            return ""

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": os.getenv("YOUR_SITE_URL", ""),  # Optional, for including your app on openrouter.ai rankings
            "X-Title": os.getenv("YOUR_SITE_NAME", "")  # Optional, shows in rankings on openrouter.ai
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

        print_debug(f"Final messages structure: {messages}")

        body = {
            "model": model_key,
            "messages": messages,
            "max_tokens": 4000,
            "temperature": 0.7
        }
        print_debug(f"Request body prepared: {body}")
        print_api_request("Sending request to Openrouter")
        print_api_request(f"Model: {model_key}")
        print_api_request(f"System prompt: {system_prompt}\n")
        print_api_request(f"User prompt: {user_prompt}\n")
        if image_data:
            print_api_request(f"Image data included: {len(image_data) if isinstance(image_data, list) else 1} image(s)")

        try:
            print_debug("Sending POST request to Openrouter API")
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
            print_debug(f"Response status code: {response.status_code}")
            response.raise_for_status()  # Raises an HTTPError for bad responses
            response_data = response.json()
            print_debug(f"Response data: {response_data}")
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                generated_text = response_data['choices'][0]['message']['content']
                print_api_response(generated_text.strip())
                print_debug(f"Generated text: {generated_text.strip()}")
                return generated_text.strip()
            else:
                print_error(f"Unexpected response format. 'choices' key not found or empty.")
                print_error(f"Response data: {response_data}")
                return ""
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            print_debug(f"HTTP error occurred. Status code: {status_code}")
            if status_code == 400:
                print_error("Bad Request: Invalid or missing parameters. Check your input and try again.")
            elif status_code == 401:
                print_error("Unauthorized: Invalid API key or expired OAuth session. Check your credentials.")
            elif status_code == 402:
                print_error("Payment Required: Insufficient credits. Please add more credits to your account.")
            elif status_code == 403:
                print_error("Forbidden: Your input was flagged by content moderation.")
            elif status_code == 408:
                print_error("Request Timeout: Your request took too long to process. Try again or simplify your input.")
            elif status_code == 429:
                print_error("Too Many Requests: You are being rate limited. Please slow down your requests.")
            elif status_code == 502:
                print_error("Bad Gateway: The chosen model is currently unavailable. Try again later or use a different model.")
            elif status_code == 503:
                print_error("Service Unavailable: No available model provider meets your routing requirements.")
            else:
                print_error(f"HTTP error occurred: {e}")
            print_error(f"Response content: {e.response.content}")
            print_debug(f"Full exception details: {e}")
            return ""
        except requests.exceptions.RequestException as e:
            print_error(f"Request error occurred: {e}")
            print_error(f"Response content: {e.response.content if e.response else 'No response content'}")
            print_debug(f"Full exception details: {e}")
            return ""
        except KeyError as e:
            print_error(f"Key error occurred: {e}")
            print_error(f"Response data: {response_data}")
            print_debug(f"Full exception details: {e}")
            return ""
        except Exception as e:
            print_error(f"An unexpected error occurred: {e}")
            print_debug(f"Full exception details: {e}")
            return ""

    @staticmethod
    def haiku(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3-haiku", system_prompt, user_prompt, image_data)

    @staticmethod
    def sonnet(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3-sonnet", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def sonnet_3_5(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3.5-sonnet", system_prompt, user_prompt, image_data)

    @staticmethod
    def opus(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("anthropic/claude-3-opus", system_prompt, user_prompt, image_data)

    @staticmethod
    def gpt_3_5_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("openai/gpt-3.5-turbo", system_prompt, user_prompt, image_data)

    @staticmethod
    def gpt_4_turbo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4-turbo", system_prompt, user_prompt, image_data)

    @staticmethod
    def gpt_4(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4", system_prompt, user_prompt, image_data)

    @staticmethod
    def gpt_4o(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4o", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def gpt_4o_mini(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("openai/gpt-4o-mini", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def gemini_flash_1_5(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("google/gemini-flash-1.5", system_prompt, user_prompt, image_data)

    @staticmethod
    def llama_3_70b_sonar_32k(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("perplexity/llama-3-sonar-large-32k-chat", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def command_r(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("cohere/command-r-plus", system_prompt, user_prompt, image_data)

    @staticmethod
    def nous_hermes_2_mistral_7b_dpo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("nousresearch/nous-hermes-2-mistral-7b-dpo", system_prompt, user_prompt, image_data)

    @staticmethod
    def nous_hermes_2_mixtral_8x7b_dpo(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("nousresearch/nous-hermes-2-mixtral-8x7b-dpo", system_prompt, user_prompt, image_data)

    @staticmethod
    def nous_hermes_yi_34b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("nousresearch/nous-hermes-yi-34b", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def qwen_2_72b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("qwen/qwen-2-72b-instruct", system_prompt, user_prompt, image_data)

    @staticmethod
    def mistral_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("mistralai/mistral-7b-instruct", system_prompt, user_prompt, image_data)

    @staticmethod
    def mistral_7b_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("mistralai/mistral-7b-instruct:nitro", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def mixtral_8x7b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("mistralai/mixtral-8x7b-instruct", system_prompt, user_prompt, image_data)

    @staticmethod
    def mixtral_8x7b_instruct_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("mistralai/mixtral-8x7b-instruct:nitro", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def mixtral_8x22b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("mistralai/mixtral-8x22b-instruct", system_prompt, user_prompt, image_data)

    @staticmethod
    def wizardlm_2_8x22b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("microsoft/wizardlm-2-8x22b", system_prompt, user_prompt, image_data)

    @staticmethod
    def neural_chat_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("intel/neural-chat-7b", system_prompt, user_prompt, image_data)

    @staticmethod
    def gemma_7b_it(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("google/gemma-7b-it", system_prompt, user_prompt, image_data)

    @staticmethod
    def gemini_pro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("google/gemini-pro", system_prompt, user_prompt, image_data)

    @staticmethod
    def llama_3_8b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-8b-instruct", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def llama_3_70b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-70b-instruct", system_prompt, user_prompt, image_data)
        
    @staticmethod
    def llama_3_70b_instruct_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-70b-instruct:nitro", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def llama_3_8b_instruct_nitro(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("meta-llama/llama-3-8b-instruct:nitro", system_prompt, user_prompt, image_data)
    
    @staticmethod
    def dbrx_132b_instruct(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("databricks/dbrx-instruct", system_prompt, user_prompt, image_data)

    @staticmethod
    def deepseek_coder(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api("deepseek/deepseek-coder", system_prompt, user_prompt, image_data)

    @staticmethod
    def custom_model(custom_model: str, system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OpenrouterModels.call_openrouter_api(custom_model, system_prompt, user_prompt, image_data)


class OllamaModels:
    @staticmethod
    def call_ollama(model: str, system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        print_debug(f"Entering call_ollama function")
        print_debug(f"Parameters: model={model}, system_prompt={system_prompt}, user_prompt={user_prompt}, image_data={image_data}")

        max_retries = 6
        base_delay = 5
        max_delay = 60

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

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
                print_api_request(f"Sending request to Ollama model: {model}")
                print_api_request(f"System prompt: '{system_prompt}'")
                print_api_request(f"User prompt: '{user_prompt}'")
                if image_data:
                    print_api_request(f"Images: {len(image_data)} included")
                else:
                    print_api_request("Images: None")

                client = ollama.Client()
                response = client.chat(model=model, messages=messages)

                response_text = response['message']['content']
                print_api_response(response_text.strip())
                print_debug(f"Returning response: {response_text.strip()}")
                return response_text.strip()

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
                    raise

            except ollama.RequestError as e:
                print_error(f"Ollama request error: {e}")
                print_debug(f"RequestError details: {e}")
                raise

            except Exception as e:
                print_error(f"An unexpected error occurred: {e}")
                print_debug(f"Unexpected error details: {type(e).__name__}, {e}")
                raise

        print_debug("Exiting call_ollama function with empty string")
        return ""

    # Llama 3 models
    @staticmethod
    def llama3_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("llama3", system_prompt, user_prompt, image_data)

    @staticmethod
    def llama3_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("llama3:70b", system_prompt, user_prompt, image_data)

    # Gemma models
    @staticmethod
    def gemma_2b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("gemma:2b", system_prompt, user_prompt, image_data)

    @staticmethod
    def gemma_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("gemma:7b", system_prompt, user_prompt, image_data)

    # Mistral model
    @staticmethod
    def mistral_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("mistral", system_prompt, user_prompt, image_data)

    # Qwen models
    @staticmethod
    def qwen_0_5b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen:0.5b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen_1_8b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen:1.8b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen_4b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen:4b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen_32b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen:32b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen_72b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen:72b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen_110b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen:110b", system_prompt, user_prompt, image_data)

    # Phi-3 models
    @staticmethod
    def phi3_3b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("phi3:3b", system_prompt, user_prompt, image_data)

    @staticmethod
    def phi3_14b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("phi3:14b", system_prompt, user_prompt, image_data)

    # Llama 2 models
    @staticmethod
    def llama2_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("llama2:7b", system_prompt, user_prompt, image_data)

    @staticmethod
    def llama2_13b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("llama2:13b", system_prompt, user_prompt, image_data)

    @staticmethod
    def llama2_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("llama2:70b", system_prompt, user_prompt, image_data)

    # CodeLlama models
    @staticmethod
    def codellama_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("codellama:7b", system_prompt, user_prompt, image_data)

    @staticmethod
    def codellama_13b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("codellama:13b", system_prompt, user_prompt, image_data)

    @staticmethod
    def codellama_34b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("codellama:34b", system_prompt, user_prompt, image_data)

    @staticmethod
    def codellama_70b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("codellama:70b", system_prompt, user_prompt, image_data)

    # Gemma 2 models
    @staticmethod
    def gemma2_9b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("gemma2:9b", system_prompt, user_prompt, image_data)

    @staticmethod
    def gemma2_27b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("gemma2:27b", system_prompt, user_prompt, image_data)

    # Qwen 2 models
    @staticmethod
    def qwen2_0_5b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen2:0.5b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen2_1_5b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen2:1.5b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen2_7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen2:7b", system_prompt, user_prompt, image_data)

    @staticmethod
    def qwen2_72b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("qwen2:72b", system_prompt, user_prompt, image_data)

    # LLaVA model
    @staticmethod
    def llava(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("llava", system_prompt, user_prompt, image_data)

    # Mixtral models
    @staticmethod
    def mixtral_8x7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("mixtral:8x7b", system_prompt, user_prompt, image_data)

    @staticmethod
    def mixtral_8x22b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("mixtral:8x22b", system_prompt, user_prompt, image_data)

    # Dolphin Mixtral models
    @staticmethod
    def dolphin_mixtral_8x7b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("dolphin-mixtral:8x7b", system_prompt, user_prompt, image_data)

    @staticmethod
    def dolphin_mixtral_8x22b(system_prompt: str, user_prompt: str, image_data: Union[List[str], str, None] = None) -> str:
        return OllamaModels.call_ollama("dolphin-mixtral:8x22b", system_prompt, user_prompt, image_data)
