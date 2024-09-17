from datetime import datetime
import base64, io
from PIL import Image

class Utils:

    @staticmethod
    def update_conversation_history(history, role, content):
        """
        Format a message with a timestamp and role, then update the conversation history.

        Args:
            history (list): The current conversation history as a list of formatted messages.
            role (str): The role of the message sender (either "User" or "Assistant").
            content (str): The message content.

        Returns:
            list: Updated conversation history.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_message = f"[{timestamp}] {role}: {content}"
        history.append(formatted_message)
        return history


    @staticmethod
    def image_to_base64(image_path: str, scale_factor: float = 0.5) -> str:
        """
        Convert an image to a base64-encoded string, with optional resizing.

        Args:
            image_path (str): The path to the image file.
            scale_factor (float, optional): Factor to scale the image dimensions. Defaults to 0.5.

        Returns:
            str: Base64-encoded string representation of the (optionally resized) image.

        Raises:
            IOError: If there's an error opening or processing the image file.
        """
        with Image.open(image_path) as img:
            # Calculate new dimensions
            new_width = int(img.width * scale_factor)
            new_height = int(img.height * scale_factor)
            
            # Resize the image
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save the resized image to a bytes buffer
            buffer = io.BytesIO()
            resized_img.save(buffer, format="PNG")
            
            # Encode the buffer to base64
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return encoded_string
