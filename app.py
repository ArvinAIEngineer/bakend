import os
import json
import re
import logging
import hashlib
from typing import Dict, Tuple, List, Optional, Any
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from fuzzywuzzy import fuzz
from supabase import create_client, Client
from asgiref.wsgi import WsgiToAsgi # For Uvicorn

# --- Basic Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, # Set to INFO for production if DEBUG is too verbose
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler() # Outputs logs to console
        # You could add logging.FileHandler("app.log") here to log to a file
    ]
)
logger = logging.getLogger(__name__)

# --- API Keys & Config ---
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_ANON_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile") # Default value if not set
FUZZY_MATCH_THRESHOLDS = {
    "name": int(os.getenv("FUZZY_NAME_THRESHOLD", "80")),
    "company": int(os.getenv("FUZZY_COMPANY_THRESHOLD", "75")),
    "email": int(os.getenv("FUZZY_EMAIL_THRESHOLD", "95")), # Email should be a high match
    "phone": int(os.getenv("FUZZY_PHONE_THRESHOLD", "95"))  # Phone also high match
}

# --- Validate Environment & Initialize Clients ---
missing_vars = []
if not NANONETS_API_KEY: missing_vars.append("NANONETS_API_KEY")
if not GROQ_API_KEY: missing_vars.append("GROQ_API_KEY")
if not SUPABASE_URL: missing_vars.append("VITE_SUPABASE_URL")
if not SUPABASE_KEY: missing_vars.append("VITE_SUPABASE_ANON_KEY")

if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}. Some features might not work.")

groq_client: Optional[Groq] = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        logger.info("Supabase client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}")

# --- Utility Functions ---
def respond(
    message: str,
    status: str = "success",
    code: int = 200,
    **additional_data
) -> Tuple[Dict[str, Any], int]:
    response_data = {"message": message, "status": status}
    response_data.update(additional_data)
    return jsonify(response_data), code

def clean_phone_number(phone: Optional[str]) -> str:
    if not phone:
        return ""
    # Keep '+' for international numbers, remove all other non-digits
    return re.sub(r'[^\d+]', '', str(phone))

def clean_identifier(value: Optional[str], field_type: str = "generic") -> Optional[str]:
    if value is None:
        return None
    s_value = str(value).strip()
    if not s_value: # If after stripping, it's an empty string
        return None

    if field_type == "gst_number" or field_type == "pan_number":
        return s_value.upper() # Ensure these are uppercase
    elif field_type == "phone_number":
        return clean_phone_number(s_value)
    elif field_type == "email_address":
        return s_value.lower() # Standardize email to lowercase
    # For other fields like 'name', 'company', 'address', just return stripped value
    return s_value

# --- Text Processing Functions ---
def extract_text_from_image(image_file) -> Tuple[Optional[str], Optional[str]]:
    if not NANONETS_API_KEY:
        logger.error("Nanonets API key not configured for OCR.")
        return None, "Nanonets API key not configured"

    nanonets_url = "https://app.nanonets.com/api/v2/OCR/FullText"
    try:
        logger.info(f"Sending image to Nanonets OCR: {image_file.filename}")
        image_file.seek(0)  # Ensure stream is at the beginning
        image_bytes = image_file.read()
        image_file.seek(0)  # Reset stream position in case it's used again (good practice)

        # Log details about the file being sent
        md5_hash = hashlib.md5(image_bytes).hexdigest()
        logger.debug(f"Image details for Nanonets: filename='{image_file.filename}', size={len(image_bytes)} bytes, md5='{md5_hash}'")

        # Nanonets expects the file content type, fallback if not available
        content_type = image_file.content_type or 'application/octet-stream' # More generic fallback
        
        files = {'file': (image_file.filename, image_bytes, content_type)}
        
        response = requests.post(
            nanonets_url,
            auth=requests.auth.HTTPBasicAuth(NANONETS_API_KEY, ''),
            files=files,
            timeout=60  # Increased timeout to 60 seconds for larger/slower OCR
        )

        logger.debug(f"Nanonets API response status: {response.status_code}")
        # --- ADDED DEBUG LINE FOR RAW TEXT RESPONSE ---
        # Log the first 1000 characters of the raw text response to avoid overly long logs
        logger.debug(f"Nanonets raw response text (first 1000 chars): {response.text[:1000]}")
        # --- END OF ADDED DEBUG LINE ---

        try:
            response_json = response.json()
            # Log the full Nanonets response (parsed as JSON) for thorough debugging
            logger.debug(f"Nanonets response parsed JSON: {json.dumps(response_json, indent=2)}")
        except json.JSONDecodeError:
            # If it's not JSON, the raw text log above would have captured it.
            # This error log is still useful to indicate the parsing failure.
            logger.error(f"Nanonets API returned non-JSON response (status {response.status_code}). Raw text already logged above. Error while parsing: {response.text[:500]}")
            return None, f"OCR service error: Invalid JSON response (status {response.status_code})"

        if response.status_code != 200:
            error_message = response_json.get("message", response.text) # Prefer message from JSON if available
            logger.error(f"Nanonets API returned status {response.status_code}: {error_message}")
            return None, f"OCR service error: {error_message} (status {response.status_code})"

        # --- TEXT EXTRACTION LOGIC ALIGNED WITH STREAMLIT ---
        all_extracted_text_parts = []
        def find_raw_text(obj):
            if isinstance(obj, dict):
                if "raw_text" in obj and isinstance(obj["raw_text"], str):
                    all_extracted_text_parts.append(obj["raw_text"])
                for value in obj.values():
                    find_raw_text(value)
            elif isinstance(obj, list):
                for item in obj:
                    find_raw_text(item)
        
        find_raw_text(response_json)
        
        if not all_extracted_text_parts:
            logger.warning("No 'raw_text' field found in Nanonets JSON structure. Nanonets message: %s", response_json.get("message"))
            nanonets_msg_detail = response_json.get("message", "No 'raw_text' field in response.")
            if "result" in response_json and isinstance(response_json["result"], list) and not response_json["result"]:
                nanonets_msg_detail = "Nanonets processed the image but found no text elements."
            return "", f"No text could be extracted from the image. (Details: {nanonets_msg_detail})"

        full_raw_text = "\n".join(all_extracted_text_parts).strip()
        # -----------------------------------------------------------------

        if not full_raw_text: # Check if, after joining, the text is still empty
            logger.warning("Extracted text from Nanonets is empty after joining parts.")
            return "", "No text content found in the image after OCR processing."

        # Ensure text is UTF-8 safe
        safe_text = full_raw_text.encode('utf-8', 'replace').decode('utf-8')
        logger.info(f"Nanonets: Successfully extracted text (length: {len(safe_text)})")
        return safe_text, None

    except requests.exceptions.Timeout:
        logger.error("Nanonets API request timed out.")
        return None, "OCR request timed out. The image might be too large or the service is slow."
    except requests.exceptions.RequestException as e:
        logger.error(f"Nanonets API request failed: {e}")
        error_detail = str(e)
        if e.response is not None:
            try: # Try to get a more specific message from Nanonets error response
                error_content = e.response.json()
                error_detail = error_content.get("message", json.dumps(error_content))
            except json.JSONDecodeError:
                error_detail = e.response.text[:500] # Log first 500 chars of non-JSON error
        return None, f"OCR service connection error: {error_detail}"
    except Exception as e:
        logger.error(f"Unexpected error in Nanonets OCR processing: {e}", exc_info=True)
        return None, f"An unexpected error occurred while processing the image with OCR: {str(e)}"

def extract_entities_with_groq(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not groq_client:
        logger.error("Groq client not initialized, cannot extract entities.")
        return None, "Groq client not initialized"
    if not text or not text.strip():
        logger.debug("No text provided for Groq entity extraction.")
        # Return an empty dict structure rather than None if no text, as per prompt expectation
        return {
            "name": None, "phone_number": None, "email_address": None, 
            "company": None, "gst_number": None, "pan_number": None, "address": None
        }, "No text provided for entity extraction"

    logger.info(f"Sending text to Groq (model: {GROQ_MODEL}) for entity extraction (length: {len(text)})")
    system_prompt = """
    You are an AI assistant specialized in accurately extracting specific fields from unstructured text, often from business cards, contact details, or user messages. Extract the following fields if present, mapping them to these exact JSON keys, which correspond to database columns:
    - "name": The full name of the primary individual.
    - "phone_number": The primary phone number. Normalize by removing spaces, hyphens, parentheses, keeping only digits and a leading '+' if applicable.
    - "email_address": The primary email address, in lowercase.
    - "company": The name of the company or organization.
    - "gst_number": The GST Identification Number (GSTIN), in uppercase.
    - "pan_number": The Permanent Account Number (PAN), in uppercase.
    - "address": The full mailing address, combined into a single string if multi-line.

    Return ONLY a valid JSON object containing these fields. Use null for fields that are not found or cannot be confidently extracted. Prioritize accuracy. Clean up common prefixes like 'Tel:', 'Email:', 'GSTIN:', 'PAN No.:', etc. If multiple phone numbers or emails are present, select the primary one based on context.
    For example, if the text is "Meet John Doe of ACME Corp, GST 123, PAN ABC, call 9999988888 or email john@acme.com", the output should be:
    {
        "name": "John Doe",
        "phone_number": "9999988888",
        "email_address": "john@acme.com",
        "company": "ACME Corp",
        "gst_number": "123",
        "pan_number": "ABC",
        "address": null
    }
    If the text is empty or contains no extractable entities, return:
    {
        "name": null,
        "phone_number": null,
        "email_address": null,
        "company": null,
        "gst_number": null,
        "pan_number": null,
        "address": null
    }
    """
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            model=GROQ_MODEL,
            temperature=0.1, # Low temperature for factual extraction
            max_tokens=1500, # Sufficient for JSON output
            response_format={"type": "json_object"} # Request JSON output
        )
        result_content = chat_completion.choices[0].message.content
        logger.debug(f"Groq raw response: {result_content}")
        extracted_data_raw = json.loads(result_content)

        # Clean the extracted data
        cleaned_data: Dict[str, Optional[str]] = {}
        expected_keys = ['name', 'phone_number', 'email_address', 'company', 'gst_number', 'pan_number', 'address']

        for key in expected_keys:
            raw_value = extracted_data_raw.get(key) # Use .get() to handle missing keys gracefully
            cleaned_data[key] = clean_identifier(raw_value, key)

        logger.info(f"Groq: Successfully extracted and cleaned structured data: {json.dumps(cleaned_data)}")
        return cleaned_data, None

    except json.JSONDecodeError as e:
        logger.error(f"Groq: Failed to parse response as JSON: {e}. Response: {result_content if 'result_content' in locals() else 'N/A'}")
        return None, "Failed to parse extracted data from LLM."
    except Exception as e:
        logger.error(f"Groq: Error in entity extraction: {e}", exc_info=True)
        return None, f"LLM entity extraction failed: {str(e)}"

# --- Database Operations ---
def get_all_customers() -> List[Dict[str, Any]]: # Not async, Supabase Python client is sync by default
    if not supabase:
        logger.error("Supabase client not available for get_all_customers")
        return []
    try:
        response = supabase.table('customers').select(
            'id, name, phone_number, email_address, company, gst_number, pan_number, address' # Specify columns
        ).order('name').execute() # Order by name for consistency
        
        # Supabase client v1.x returns PostgrestAPIResponse, .data is the list
        # Supabase client v2.x might change this, check documentation if upgrading
        return response.data if hasattr(response, 'data') else []
    except Exception as e:
        logger.error(f"Supabase: Failed to fetch all customers: {e}", exc_info=True)
        return []

async def search_customers_in_db(query_term: str) -> List[Dict[str, Any]]:
    if not supabase:
        logger.error("Supabase client not available for search_customers_in_db")
        return []
    
    logger.info(f"Supabase: Searching for customers with query: '{query_term}'")
    
    # Clean query terms for specific fields
    clean_query_gst = clean_identifier(query_term, "gst_number")
    clean_query_pan = clean_identifier(query_term, "pan_number")
    clean_query_email = clean_identifier(query_term, "email_address")
    # For phone, we need to be careful as DB might not store it normalized
    # We'll do an exact match on cleaned phone later if other searches fail.
    # For now, use the raw query_term for phone if it looks like one, or do a broader search.

    all_results: Dict[int, Dict[str, Any]] = {} # Use dict to avoid duplicates by ID

    # Helper for executing Supabase queries
    # Supabase Python client's execute() is blocking, so no `await` needed directly on it
    # The `async def` is for Flask's async route handling, not for Supabase client calls.
    def execute_and_store_sync(field_name, value_to_match, match_type="eq"):
        if not value_to_match: return
        try:
            if match_type == "ilike":
                db_response = supabase.table('customers').select('*').ilike(field_name, f'%{value_to_match}%').execute()
            else: # eq
                db_response = supabase.table('customers').select('*').eq(field_name, value_to_match).execute()
            
            if db_response.data:
                logger.info(f"Supabase: Found {len(db_response.data)} by {field_name} {match_type} '{value_to_match}'")
                for item in db_response.data: all_results[item['id']] = item
        except Exception as e:
            logger.error(f"Supabase: Error searching by {field_name} for '{value_to_match}': {e}")

    # Exact matches first (more reliable)
    if clean_query_gst: execute_and_store_sync('gst_number', clean_query_gst)
    if not all_results and clean_query_pan: execute_and_store_sync('pan_number', clean_query_pan)
    # Email match (case-insensitive exact usually best)
    if not all_results and clean_query_email: execute_and_store_sync('email_address', clean_query_email, match_type="eq") # use eq for email after cleaning

    # Phone number search: requires client-side normalization and comparison
    # as DB might store phones with varied formatting.
    clean_query_phone = clean_phone_number(query_term)
    if not all_results and clean_query_phone:
        all_cust_for_phone = get_all_customers() # This is a sync call
        for cust in all_cust_for_phone:
            db_phone_cleaned = clean_phone_number(cust.get('phone_number'))
            if db_phone_cleaned and db_phone_cleaned == clean_query_phone:
                all_results[cust['id']] = cust
                logger.info(f"Supabase: Found by normalized phone client-side for '{query_term}' matching '{clean_query_phone}'")
                # break # Found one, usually phone is unique enough (removed break to catch all matches)

    # Broader ilike searches if no exact matches found
    if not all_results: execute_and_store_sync('name', query_term, match_type="ilike")
    if not all_results: execute_and_store_sync('company', query_term, match_type="ilike")
    # Broader email search if exact email didn't match
    if not all_results and clean_query_email: execute_and_store_sync('email_address', clean_query_email, match_type="ilike") 


    # Fuzzy matching as a last resort if still no results and query is reasonably long
    if not all_results and len(query_term) > 2:
        logger.info(f"Supabase: No exact/ilike matches for '{query_term}', trying fuzzy on all customers.")
        all_customers_for_fuzzy = get_all_customers() # Sync call
        fuzzy_scored_matches = []
        for cust in all_customers_for_fuzzy:
            # Score against multiple fields
            name_score = fuzz.token_set_ratio(query_term.lower(), (cust.get('name') or "").lower())
            company_score = fuzz.token_set_ratio(query_term.lower(), (cust.get('company') or "").lower())
            
            # Prioritize name matches for general queries
            if name_score > FUZZY_MATCH_THRESHOLDS["name"]:
                fuzzy_scored_matches.append((name_score, cust))
            elif company_score > FUZZY_MATCH_THRESHOLDS["company"]:
                 fuzzy_scored_matches.append((company_score, cust)) # Could use a slightly lower threshold for company

        if fuzzy_scored_matches:
            fuzzy_scored_matches.sort(key=lambda x: x[0], reverse=True) # Sort by score desc
            for score, cust in fuzzy_scored_matches[:5]: # Take top 5 fuzzy matches
                 if cust['id'] not in all_results: # Add if not already found
                    all_results[cust['id']] = cust
            logger.info(f"Supabase: Added up to {min(5, len(fuzzy_scored_matches))} fuzzy matches for '{query_term}'")
            
    return list(all_results.values())

async def find_matching_customer_in_db(extracted_entities: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not supabase or not extracted_entities:
        logger.warning("Supabase client or extracted entities missing for find_matching_customer_in_db")
        return False, None
    logger.info(f"Supabase: Finding match for extracted entities: {json.dumps(extracted_entities)}")

    # Check for exact matches on unique identifiers first
    unique_checks = [
        ('gst_number', clean_identifier(extracted_entities.get('gst_number'), 'gst_number')),
        ('pan_number', clean_identifier(extracted_entities.get('pan_number'), 'pan_number')),
        ('email_address', clean_identifier(extracted_entities.get('email_address'), 'email_address')),
    ]
    for field, value in unique_checks:
        if value: # Only query if value is not None/empty after cleaning
            try:
                op = 'eq' # Use 'eq' for exact matches
                # For email, Supabase stores it as text, so 'eq' on the cleaned lowercase value is appropriate.
                # If DB stores mixed case emails, then 'ilike' would be needed, but we clean to lowercase.
                
                response = supabase.table('customers').select('*').filter(field, op, value).limit(1).execute()
                if response.data:
                    logger.info(f"Supabase: Exact match found on '{field}' for value '{value}'")
                    return True, response.data[0]
            except Exception as e:
                logger.error(f"Supabase: Error checking exact match for {field}='{value}': {e}")

    # Check for phone number match (requires normalization)
    ex_phone_clean = clean_identifier(extracted_entities.get('phone_number'), 'phone_number')
    if ex_phone_clean:
        all_customers = get_all_customers() # Sync call
        for customer in all_customers:
            db_phone_clean = clean_phone_number(customer.get('phone_number'))
            if db_phone_clean and db_phone_clean == ex_phone_clean:
                logger.info(f"Supabase: Exact normalized phone match found for '{ex_phone_clean}'")
                return True, customer

    # Fuzzy matching if no exact match is found
    all_customers_for_fuzzy = get_all_customers() # Sync call
    if not all_customers_for_fuzzy:
        logger.info("Supabase: No customers found in database for fuzzy matching.")
        return False, None

    fuzzy_fields = ['name', 'company'] 

    for field in fuzzy_fields:
        ex_value = clean_identifier(extracted_entities.get(field)) 
        if not ex_value:
            continue 
        
        ex_value_lower = ex_value.lower()

        for customer in all_customers_for_fuzzy:
            db_value = customer.get(field) or "" 
            if not db_value:
                continue 

            score = fuzz.token_set_ratio(ex_value_lower, db_value.lower())
            threshold = FUZZY_MATCH_THRESHOLDS.get(field, 75) 
            
            if score >= threshold:
                logger.info(f"Supabase: Fuzzy match found on '{field}' for '{ex_value}' with score {score} (threshold {threshold}) (customer: {customer.get('name')})")
                return True, customer 

    logger.info("Supabase: No sufficiently strong unique or fuzzy match found.")
    return False, None

async def add_customer_to_db(customer_data: Dict[str, Any]) -> Dict[str, Any]:
    if not supabase:
        logger.error("Database client (Supabase) not available for adding customer.")
        return {"status": "error", "message": "Database client not available"}
    
    logger.info(f"Supabase: Attempting to add customer: {customer_data.get('name', 'N/A')}")

    data_to_insert: Dict[str, Optional[str]] = {}
    valid_supabase_columns = ['name', 'phone_number', 'email_address', 'company', 'gst_number', 'pan_number', 'address']
    for col in valid_supabase_columns:
        raw_value = customer_data.get(col)
        data_to_insert[col] = clean_identifier(raw_value, col) 

    if not data_to_insert.get('name'):
        logger.warning("Supabase: Add customer failed - 'name' is missing or empty after cleaning.")
        return {"status": "error", "message": "Customer name is required and cannot be empty."}

    try:
        response = supabase.table('customers').insert(data_to_insert).execute()

        if response.data and len(response.data) > 0:
            new_customer_record = response.data[0]
            logger.info(f"Supabase: Successfully added customer ID {new_customer_record.get('id')}: {new_customer_record.get('name')}")
            return {"status": "success", "message": "Customer added successfully.", "customer_data": new_customer_record}
        else:
            error_info = getattr(response, 'error', None) 
            if error_info: 
                logger.error(f"Supabase: Insert failed with Supabase error: {error_info}")
                msg = error_info.message if hasattr(error_info, 'message') else str(error_info)
                
                if "duplicate key value violates unique constraint" in msg.lower():
                    field_match = re.search(r"constraint \"customers_(\w+)_key\"", msg) 
                    field = field_match.group(1).replace('_', ' ') if field_match else "a unique field (e.g., GST, PAN, Email)"
                    return {"status": "error", "message": f"A customer with this {field} already exists."}
                return {"status": "error", "message": f"Database error: {msg}"}
            
            logger.warning(f"Supabase: Insert command executed but no data returned and no explicit error. Response: {response}")
            return {"status": "error", "message": "Customer may have been added, but confirmation from database failed."}

    except Exception as e: 
        logger.error(f"Supabase: Exception during customer insertion: {e}", exc_info=True)
        if "duplicate key value violates unique constraint" in str(e).lower():
             field_match = re.search(r"constraint \"customers_(\w+)_key\"", str(e))
             field = field_match.group(1).replace('_', ' ') if field_match else "a unique field"
             return {"status": "error", "message": f"A customer with this {field} already exists (from exception)."}
        return {"status": "error", "message": f"Failed to add customer due to a server error: {str(e)}"}

# --- API Routes (Async for Flask if using an ASGI server like Uvicorn) ---
@app.route('/api/chat', methods=['POST'])
async def handle_chat_route():
    if not request.is_json:
        return respond("Request must be JSON", "error", 415) # Unsupported Media Type
    
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return respond("No message provided", "error", 400) # Bad Request

    logger.info(f"Chat: Received message: '{user_message}'")

    if user_message.lower() in ['hi', 'hello', 'hey', 'start', 'greetings', 'good morning', 'good evening']:
        return respond(
            "Hello! I'm JIA, your Customer Database Assistant. How can I help? You can ask me to:\n"
            " • 'Find customer [name/GST/email/etc.]'\n"
            " • 'Check details for [customer information]'\n"
            " • Or upload a visiting card.",
            status="greeting"
        )

    search_match = re.match(r"(?:find|search|lookup|get|show)\s+(?:customer|client|contact|details\s+for)?\s*(.+)", user_message, re.IGNORECASE)
    if search_match:
        query = search_match.group(1).strip()
        if not query: 
            return respond("Please specify who or what you are looking for (e.g., name, GST, email).", status="query_missing", code=400)
        
        logger.info(f"Chat: Search intent identified. Query: '{query}'")
        try:
            found_customers = await search_customers_in_db(query)
        except Exception as e:
            logger.error(f"Chat: Error calling search_customers_in_db: {e}", exc_info=True)
            return respond("An error occurred while searching for customers.", "error", 500)

        if not found_customers:
            return respond(f"I couldn't find any customers matching '{query}'. You can try different details or add them as a new customer.", status="not_found")
        if len(found_customers) == 1:
            cust = found_customers[0]
            details_str = f"Name: {cust.get('name', 'N/A')}\n" \
                          f"Company: {cust.get('company', 'N/A')}\n" \
                          f"GST: {cust.get('gst_number', 'N/A')}\n" \
                          f"PAN: {cust.get('pan_number', 'N/A')}\n" \
                          f"Email: {cust.get('email_address', 'N/A')}\n" \
                          f"Phone: {cust.get('phone_number', 'N/A')}\n" \
                          f"Address: {cust.get('address', 'N/A')}"
            return respond(
                f"Found customer:\n{details_str}",
                status="found_single", customer_data=cust
            )
        else: 
            names = [f"• {c.get('name', 'Unknown')} (Company: {c.get('company', 'N/A')}, GST: {c.get('gst_number', 'N/A')})" for c in found_customers[:5]] 
            return respond(
                f"Found multiple potential matches for '{query}':\n" + "\n".join(names) + "\n\nPlease be more specific or provide more details.",
                status="found_multiple", customer_data=found_customers[:5] 
            )

    logger.info(f"Chat: No simple search intent, using LLM for entity extraction from: '{user_message}'")
    try:
        extracted_entities, extraction_error = extract_entities_with_groq(user_message)
    except Exception as e:
        logger.error(f"Chat: Error calling extract_entities_with_groq: {e}", exc_info=True)
        return respond("There was an issue processing your message with the AI model.", "error", 500)

    if extraction_error:
        return respond(f"Sorry, I had trouble understanding the details: {extraction_error}", status="extraction_error", code=400 if "No text" in extraction_error else 500)
    
    if not extracted_entities or not any(v for v in extracted_entities.values()): 
        return respond("I couldn't extract specific customer details from your message. Could you try rephrasing or providing more information?", status="extraction_failed")

    logger.info(f"Chat: LLM extracted: {json.dumps(extracted_entities)}")
    
    try:
        match_found, existing_customer_data = await find_matching_customer_in_db(extracted_entities)
    except Exception as e:
        logger.error(f"Chat: Error calling find_matching_customer_in_db: {e}", exc_info=True)
        return respond("An error occurred while checking against the existing customer database.", "error", 500)

    if match_found and existing_customer_data:
        details_str = f"Name: {existing_customer_data.get('name', 'N/A')}\n" \
                      f"Company: {existing_customer_data.get('company', 'N/A')}\n" \
                      f"GST: {existing_customer_data.get('gst_number', 'N/A')}"
        return respond(
            f"This looks like an existing customer: {details_str}.\nIs this the correct one, or would you like to add a new entry with the details you provided?",
            status="existing_customer_chat", 
            customer_data=existing_customer_data, 
            extracted_data=extracted_entities 
        )
    else:
        details_summary = "\n".join([f"  • {key.replace('_', ' ').title()}: {value}" for key, value in extracted_entities.items() if value])
        return respond(
            f"I've extracted these details from your message:\n{details_summary}\n\nThis seems to be a new customer. Would you like me to add them, or would you like to use the form to confirm/edit?",
            status="new_potential_chat", 
            extracted_data=extracted_entities
        )

@app.route('/api/upload-card', methods=['POST'])
async def handle_card_upload_route():
    if 'card' not in request.files:
        logger.error("CardUpload: 'card' file part missing in request.")
        return respond("No 'card' file part in the request.", "error", 400)
    
    file = request.files['card']
    if file.filename == '':
        logger.error("CardUpload: No file selected for upload (empty filename).")
        return respond("No file selected for upload.", "error", 400)

    if not file: 
        logger.error("CardUpload: File object is None.")
        return respond("File not provided correctly.", "error", 400)
        
    logger.info(f"CardUpload: Received card for processing: {file.filename}")

    raw_text = "" # Initialize raw_text
    extracted_entities = {} # Initialize extracted_entities

    try:
        # 1. Extract text from image using Nanonets
        raw_text, ocr_error = extract_text_from_image(file)

        if ocr_error: 
            raw_text_safe = raw_text if raw_text else ""
            logger.warning(f"CardUpload: OCR failed or returned an error. OCR Error: '{ocr_error}', Raw Text (if any): '{raw_text_safe}'")
            return respond(
                f"Could not extract text from the card. {ocr_error}", 
                "extraction_failed_card", 
                code=400 if "key not configured" in ocr_error else 500, 
                raw_text=raw_text_safe, 
                extracted_data={} 
            )
        
        if not raw_text: 
            logger.info("CardUpload: OCR processed, but no text content was found in the image.")
            return respond(
                "OCR processed the image, but no text content was found. Please try a clearer image or enter details manually.",
                "extraction_failed_card",
                code=200, 
                raw_text="",
                extracted_data={}
            )

        logger.info(f"CardUpload: OCR successful. Extracted text (first 100 chars): {raw_text[:100]}...")

        # 2. Extract structured entities using Groq
        extracted_entities, llm_error = extract_entities_with_groq(raw_text)

        if llm_error:
            logger.error(f"CardUpload: LLM entity extraction failed: {llm_error}")
            return respond(
                f"Entity extraction from card's text failed: {llm_error}",
                "llm_error_card", 
                code=500,
                raw_text=raw_text, 
                extracted_data={}
            )
        
        if not extracted_entities or not any(v for v in extracted_entities.values()):
            logger.info("CardUpload: LLM processed text, but no structured entities were extracted.")
            return respond(
                "Could not extract structured details from the card's text. You can review the raw text and add details manually.",
                "extraction_failed_card", 
                code=200, 
                raw_text=raw_text,
                extracted_data=extracted_entities if extracted_entities else {} 
            )

        logger.info(f"CardUpload: LLM extracted from card: {json.dumps(extracted_entities)}")

        # 3. Find matching customer in DB
        match_found, existing_customer_data = await find_matching_customer_in_db(extracted_entities)

        if match_found and existing_customer_data:
            logger.info(f"CardUpload: Matched existing customer: {existing_customer_data.get('name')}")
            return respond(
                "Card processed. This appears to be an existing customer.",
                status="existing_customer_card",
                matched_customer=existing_customer_data,
                extracted_data=extracted_entities,
                raw_text=raw_text
            )
        else:
            logger.info("CardUpload: No match found for extracted details, treating as new potential customer.")
            return respond(
                "Card processed. Details extracted for a new potential customer. Please review and confirm.",
                status="new_customer_card",
                extracted_data=extracted_entities,
                raw_text=raw_text,
                matched_customer=None 
            )

    except Exception as e: 
        logger.error(f"CardUpload: Unexpected error in /api/upload-card route: {e}", exc_info=True)
        # Use the initialized or potentially populated variables for raw_text and extracted_entities
        return respond(
            f"A critical error occurred while processing the card: {str(e)}",
            "error", 
            500,
            raw_text=raw_text, 
            extracted_data=extracted_entities
        )

@app.route('/api/customers', methods=['POST'])
async def add_new_customer_route():
    if not request.is_json:
        return respond("Request must be JSON", "error", 415)
    
    customer_data_from_frontend = request.json
    if not customer_data_from_frontend: 
        return respond("No customer data provided in the request.", "error", 400)

    logger.info(f"AddCustomerRoute: Received data for new customer: {json.dumps(customer_data_from_frontend)}")
    
    try:
        add_result = await add_customer_to_db(customer_data_from_frontend)
        
        if add_result["status"] == "success":
            return respond(
                add_result["message"], 
                status="success", 
                code=201, 
                customer_data=add_result.get("customer_data")
            )
        else:
            error_code = 409 if "already exists" in add_result["message"].lower() else 400 
            return respond(add_result["message"], status="error", code=error_code)
            
    except Exception as e:
        logger.error(f"AddCustomerRoute: Critical error calling add_customer_to_db: {e}", exc_info=True)
        return respond("A server error occurred while trying to add the customer.", "error", 500)

@app.route('/api/customers', methods=['GET'])
async def get_customers_route():
    query = request.args.get('query') 
    
    try:
        if query:
            logger.info(f"GetCustomersRoute: Searching customers with query: '{query}'")
            customers = await search_customers_in_db(query)
            return respond(f"Found {len(customers)} customers matching query.", "success", customers=customers)
        else:
            logger.info("GetCustomersRoute: Fetching all customers (no query specified).")
            customers = get_all_customers() 
            return respond(f"Retrieved {len(customers)} customers.", "success", customers=customers)
            
    except Exception as e:
        logger.error(f"GetCustomersRoute: Error during customer retrieval/search: {e}", exc_info=True)
        return respond("Failed to retrieve or search customers due to a server error.", "error", 500)

@app.route('/health', methods=['GET'])
async def health_check_route():
    status = "ok"
    services = {
        "api_status": "running",
        "supabase_client": "initialized" if supabase else "not_initialized",
        "groq_client": "initialized" if groq_client else "not_initialized",
        "nanonets_api_key": "configured" if NANONETS_API_KEY else "not_configured"
    }

    if supabase:
        try:
            response = supabase.table('customers').select('id', count='exact').limit(0).execute()
            
            if hasattr(response, 'error') and response.error: 
                logger.error(f"HealthCheck: Supabase query error: {response.error}")
                services["supabase_db_connection"] = f"error ({str(response.error.message)[:50]}...)"
                status = "degraded" 
            elif response.count is not None: 
                 services["supabase_db_connection"] = f"connected (found {response.count} customers table entries)"
            else: 
                logger.warning(f"HealthCheck: Supabase count query returned unexpected response: {response}")
                services["supabase_db_connection"] = "query_issue_unexpected_response"
                status = "degraded"
        except Exception as e:
            logger.error(f"HealthCheck: Supabase connection exception: {e}")
            services["supabase_db_connection"] = f"exception ({str(e)[:50]}...)"
            status = "degraded"
    else:
        status = "degraded" 
        
    return respond(f"Service health status: {status}.", status, code=200 if status=="ok" else 503, services=services)

# --- ASGI Framework Adapter & Run ---
asgi_app = WsgiToAsgi(app)

if __name__ == '__main__':
    import uvicorn
    port = int(os.environ.get("PORT", 5200)) 
    should_reload = os.getenv("FLASK_ENV", "production").lower() == "development"
    
    logger.info(f"Starting Uvicorn server on http://0.0.0.0:{port} (Reload: {should_reload})")
    uvicorn.run(
        "__main__:asgi_app", # Changed from "back:asgi_app" to "__main__:asgi_app" if running this file directly
        host="0.0.0.0", 
        port=port, 
        reload=should_reload,
        log_level="debug" if should_reload else "info" 
    )
