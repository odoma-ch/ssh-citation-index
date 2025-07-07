import json
import re
import logging
from typing import Union, Any

def fix_json_formatting(json_string: str) -> str:
    """
    Fix common JSON formatting issues from LLM responses.
    
    Args:
        json_string (str): The potentially malformed JSON string
        
    Returns:
        str: Fixed JSON string
    """
    if not json_string or not json_string.strip():
        return "{}"
    
    # Remove any leading/trailing whitespace
    json_string = json_string.strip()
    
    # Remove markdown code blocks if present
    json_string = re.sub(r'^```(?:json)?\s*', '', json_string, flags=re.MULTILINE)
    json_string = re.sub(r'\s*```\s*$', '', json_string, flags=re.MULTILINE)
    
    # Fix common issues step by step
    json_string = _fix_trailing_commas(json_string)
    json_string = _fix_missing_commas(json_string)
    json_string = _fix_unquoted_keys(json_string)
    json_string = _fix_single_quotes(json_string)
    json_string = _fix_escape_sequences(json_string)
    json_string = _fix_boolean_values(json_string)
    json_string = _fix_null_values(json_string)
    json_string = _fix_unclosed_brackets(json_string)
    
    return json_string

def _fix_trailing_commas(json_string: str) -> str:
    """Remove trailing commas before closing brackets/braces."""
    # Remove trailing commas before closing braces or brackets
    json_string = re.sub(r',\s*}', '}', json_string)
    json_string = re.sub(r',\s*]', ']', json_string)
    return json_string

def _fix_missing_commas(json_string: str) -> str:
    """Add missing commas between JSON elements."""
    # Add comma after closing brace if followed by opening brace or quote
    json_string = re.sub(r'}\s*(?=["{])', '},', json_string)
    # Add comma after closing bracket if followed by opening bracket or quote
    json_string = re.sub(r']\s*(?=["\[{])', '],', json_string)
    # Add comma after quoted value if followed by quote (for object values)
    json_string = re.sub(r'"\s*(?="[^:]*":)', '",', json_string)
    # Add comma after number/boolean if followed by quote
    json_string = re.sub(r'(true|false|\d+\.?\d*)\s*(?=")', r'\1,', json_string)
    return json_string

def _fix_unquoted_keys(json_string: str) -> str:
    """Add quotes around unquoted object keys."""
    # Match unquoted keys (word characters followed by colon)
    json_string = re.sub(r'([{\s,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', json_string)
    return json_string

def _fix_single_quotes(json_string: str) -> str:
    """Convert single quotes to double quotes (but not inside strings)."""
    # This is a simplified approach - for more complex cases, you'd need a proper parser
    # Replace single quotes with double quotes, but be careful about apostrophes
    json_string = re.sub(r"'([^']*)'", r'"\1"', json_string)
    return json_string

def _fix_escape_sequences(json_string: str) -> str:
    """Fix common escape sequence issues."""
    # Fix unescaped quotes inside strings (this is tricky and may need manual review)
    # For now, we'll just ensure backslashes are properly escaped
    json_string = re.sub(r'\\(?!["\\/bfnrt])', r'\\\\', json_string)
    return json_string

def _fix_boolean_values(json_string: str) -> str:
    """Fix boolean values that might be capitalized."""
    json_string = re.sub(r'\bTrue\b', 'true', json_string)
    json_string = re.sub(r'\bFalse\b', 'false', json_string)
    return json_string

def _fix_null_values(json_string: str) -> str:
    """Fix null values that might be capitalized or use None."""
    json_string = re.sub(r'\bNull\b', 'null', json_string)
    json_string = re.sub(r'\bNone\b', 'null', json_string)
    return json_string

def _fix_unclosed_brackets(json_string: str) -> str:
    """Attempt to fix unclosed brackets/braces."""
    # Count opening and closing brackets/braces
    open_braces = json_string.count('{')
    close_braces = json_string.count('}')
    open_brackets = json_string.count('[')
    close_brackets = json_string.count(']')
    
    # Add missing closing braces
    if open_braces > close_braces:
        json_string += '}' * (open_braces - close_braces)
    
    # Add missing closing brackets  
    if open_brackets > close_brackets:
        json_string += ']' * (open_brackets - close_brackets)
    
    return json_string

def safe_json_parse(json_string: str, max_attempts: int = 3) -> Union[dict, list, None]:
    """
    Safely parse JSON string with multiple fix attempts.
    
    Args:
        json_string (str): The JSON string to parse
        max_attempts (int): Maximum number of fix attempts
        
    Returns:
        Union[dict, list, None]: Parsed JSON object or None if parsing fails
    """
    if not json_string or not json_string.strip():
        return None
    
    # First, try parsing without any fixes
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        pass
    
    # Try with fixes
    for attempt in range(max_attempts):
        try:
            fixed_json = fix_json_formatting(json_string)
            return json.loads(fixed_json)
        except json.JSONDecodeError as e:
            logging.warning(f"JSON parse attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                # For subsequent attempts, try more aggressive fixes
                json_string = _apply_aggressive_fixes(json_string)
            continue
    
    return None

def _apply_aggressive_fixes(json_string: str) -> str:
    """Apply more aggressive fixes for stubborn JSON issues."""
    # Try to extract JSON from text that might contain other content
    # Look for the first { or [ and the last } or ]
    start_brace = json_string.find('{')
    start_bracket = json_string.find('[')
    
    # Determine the starting position
    if start_brace == -1 and start_bracket == -1:
        return json_string
    elif start_brace == -1:
        start_pos = start_bracket
        end_char = ']'
    elif start_bracket == -1:
        start_pos = start_brace
        end_char = '}'
    else:
        start_pos = min(start_brace, start_bracket)
        end_char = '}' if start_pos == start_brace else ']'
    
    # Find the last occurrence of the matching closing character
    end_pos = json_string.rfind(end_char)
    
    if end_pos > start_pos:
        json_string = json_string[start_pos:end_pos + 1]
    
    return json_string

# Example usage and testing
def test_json_fixer():
    """Test the JSON fixer with various malformed JSON examples."""
    
    test_cases = [
        # Missing comma
        '{"name": "John" "age": 30}',
        # Trailing comma
        '{"name": "John", "age": 30,}',
        # Unquoted keys
        '{name: "John", age: 30}',
        # Single quotes
        "{'name': 'John', 'age': 30}",
        # Mixed issues
        "{name: 'John', age: 30, active: True}",
        # Unclosed braces
        '{"name": "John", "age": 30',
        # Array with issues
        '[{"name": "John"} {"name": "Jane"}]',
    ]
    
    for i, test_json in enumerate(test_cases):
        print(f"\nTest case {i + 1}:")
        print(f"Original: {test_json}")
        
        result = safe_json_parse(test_json)
        if result is not None:
            print(f"Fixed and parsed successfully: {result}")
        else:
            print("Failed to parse even after fixes")

if __name__ == "__main__":
    test_json_fixer()