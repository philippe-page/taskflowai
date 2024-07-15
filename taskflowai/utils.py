import tiktoken
from datetime import datetime

class Utils:
    @staticmethod
    def estimate_token_count(text, model=None):
        """
        Estimate the number of tokens in the given text using the specified model's encoding.

        This method attempts to use the encoding for the specified model. If the model is not found,
        it falls back to the 'cl100k_base' encoding.

        Args:
            text (str): The input text to estimate token count for.
            model (str, optional): The model to use for encoding. Defaults to "gpt-3.5-turbo" if None.

        Returns:
            int: The estimated number of tokens in the input text.

        Raises:
            KeyError: If the specified model is not found and the fallback encoding fails.

        Note:
            This method requires the 'tiktoken' library to be installed.
        """
        if model is None:
            model = "gpt-3.5-turbo"
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except KeyError:
            print(f"Warning: model {model} not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))

    @staticmethod
    def format_message(role, content):
        """
        Format a message with a timestamp and role.

        Args:
            role (str): The role of the message sender (either "User" or "Assistant").
            content (str): The message content.

        Returns:
            str: Formatted message with timestamp and role.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"[{timestamp}] {role}: {content}"

    @staticmethod
    def update_history(history, role, message):
        """
        Update the conversation history with the latest message.

        Args:
            history (str): The current conversation history.
            role (str): The role of the message sender (either "User" or "Assistant").
            message (str): The message content.

        Returns:
            str: Updated conversation history.
        """
        formatted_message = Utils.format_message(role, message)
        history += f"\n{formatted_message}"
        return history.strip()

    @staticmethod
    def truncate_history(history, max_tokens=8000):
        """
        Truncate the conversation history to a maximum token count, always keeping the latest messages.

        Args:
            history (str): The current conversation history.
            max_tokens (int): The maximum number of tokens to keep.

        Returns:
            str: Truncated conversation history.
        """
        messages = history.split('\n')
        total_tokens = sum(Utils.estimate_token_count(msg) for msg in messages)

        if total_tokens <= max_tokens:
            return history

        truncated_history = []
        current_tokens = 0

        for message in reversed(messages):
            message_tokens = Utils.estimate_token_count(message)
            if current_tokens + message_tokens > max_tokens:
                break
            truncated_history.append(message)
            current_tokens += message_tokens

        return '\n'.join(reversed(truncated_history))
