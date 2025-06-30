import re

class TextCleanerNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define input types for the node
        """
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "tags": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "mode": (["remove_sentences_with_tags", "remove_tags_only"], {"default": "remove_sentences_with_tags"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_text",)
    FUNCTION = "clean_text"
    CATEGORY = "🐳Pond/text"

    def parse_tags(self, tags):
        """
        Parse tags, supporting multiple separator types including Chinese and English punctuation
        """
        # Chinese and English punctuation marks
        separators = r'[,;、\n\s。，；：？！,.?!：]+' 
        
        # Split and clean tags
        tag_list = [
            tag.strip().lower() 
            for tag in re.split(separators, tags) 
            if tag.strip()
        ]
        
        return tag_list

    def clean_text(self, text, tags, mode):
        """
        Main function for cleaning text
        
        :param text: Input text
        :param tags: Tags to remove
        :param mode: Cleaning mode
        :return: Cleaned text
        """
        # If no tags provided, return original text
        if not tags:
            return (text,)

        # Parse tags
        tag_list = self.parse_tags(tags)

        # Process different cleaning modes
        if mode == "remove_tags_only":
            # Remove specific tags/prompts
            for tag in tag_list:
                # Use regex replacement, ignoring case
                text = re.sub(re.escape(tag), '', text, flags=re.IGNORECASE)
        
        elif mode == "remove_sentences_with_tags":
            # Split sentences (supporting both Chinese and English sentence delimiters)
            sentences = re.split(r'([。！？!?.]+)', text)
            
            # Reassemble sentences, excluding those containing specified tags
            filtered_sentences = []
            for i in range(0, len(sentences), 2):
                # Check if sentence contains any tags
                if i < len(sentences) and not any(tag in sentences[i].lower() for tag in tag_list):
                    # If sentence doesn't contain tags, add to result
                    filtered_sentences.append(sentences[i])
                    # If there's a delimiter after this sentence, add it too
                    if i + 1 < len(sentences):
                        filtered_sentences.append(sentences[i+1])
            
            # Reassemble text
            text = ''.join(filtered_sentences)

        # Return cleaned text
        return (text.strip(),)

# Define mappings for ComfyUI node display
NODE_CLASS_MAPPINGS = {
    "TextCleanerNode": TextCleanerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextCleanerNode": "🐳Text Cleaner"
}