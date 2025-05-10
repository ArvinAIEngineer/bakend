import os
import json
import re
import logging
import hashlib
from typing import Dict, Tuple, List, Optional, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from groq import Groq
from fuzzywuzzy import fuzz
from supabase import create_client, Client
from asgiref.wsgi import WsgiToAsgi
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import io

# --- Basic Setup ---
load_dotenv()
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- API Keys & Config ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("VITE_SUPABASE_URL")
SUPABASE_KEY = os.getenv("VITE_SUPABASE_ANON_KEY")

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
FUZZY_MATCH_THRESHOLDS = {
    "name": int(os.getenv("FUZZY_NAME_THRESHOLD", "80")),
    "company": int(os.getenv("FUZZY_COMPANY_THRESHOLD", "75")),
    "email": int(os.getenv("FUZZY_EMAIL_THRESHOLD", "95")),
    "phone": int(os.getenv("FUZZY_PHONE_THRESHOLD", "95")),
    "gst_number": int(os.getenv("FUZZY_GST_THRESHOLD", "90")),
    "pan_number": int(os.getenv("FUZZY_PAN_THRESHOLD", "90")),
    "address": int(os.getenv("FUZZY_ADDRESS_THRESHOLD", "70"))
}
FUZZY_MAX_RESULTS_CHAT = int(os.getenv("FUZZY_MAX_RESULTS_CHAT", "5"))
FUZZY_MAX_RESULTS_SEARCH = int(os.getenv("FUZZY_Max_RESULTS_SEARCH", "10"))

# --- Validate Environment & Initialize Clients ---
missing_vars = []
if not GEMINI_API_KEY: missing_vars.append("GEMINI_API_KEY")
if not GROQ_API_KEY: missing_vars.append("GROQ_API_KEY")
if not SUPABASE_URL: missing_vars.append("VITE_SUPABASE_URL")
if not SUPABASE_KEY: missing_vars.append("VITE_SUPABASE_ANON_KEY")

if missing_vars:
    logger.warning(f"Missing environment variables: {', '.join(missing_vars)}. Some features might not work.")

# Initialize Google Gemini
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Google Gemini client initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize Google Gemini client: {e}")
    genai = None

# Initialize Groq client
groq_client: Optional[Groq] = None
if GROQ_API_KEY:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq client initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")

# Initialize Supabase client
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
    return re.sub(r'[^\d+]', '', str(phone))

def clean_identifier(value: Optional[str], field_type: str = "generic") -> Optional[str]:
    if value is None:
        return None
    s_value = str(value).strip()
    if not s_value:
        return None
    if field_type == "gst_number" or field_type == "pan_number":
        return s_value.upper()
    elif field_type == "phone_number":
        return clean_phone_number(s_value)
    elif field_type == "email_address":
        return s_value.lower()
    return s_value

# --- Image Processing with Gemini ---
def extract_entities_from_image_with_gemini(image_file) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not genai:
        logger.error("Google Gemini client not configured.")
        return None, "Google Gemini client not configured"

    try:
        logger.info(f"Sending image to Google Gemini: {image_file.filename}")
        image_file.seek(0)
        image = Image.open(image_file)
        image_file.seek(0)

        prompt = """
        Here is a visiting card. Extract the Name, Company, Phone, and Email of the person.

        IMPORTANT:
        - If an attribute is missing or unclear, OMIT it rather than including placeholders like 'Not found' or 'Unknown'.
        - Ensure the Phone Number is formatted consistently (e.g., +1-123-456-7890 or 123-456-7890).
        - Validate that the Email follows a standard email format (e.g., user@domain.com).
        - Provide the output in a clean, structured JSON format with keys: name, company, phone_number, email_address.
        
        Example Output:
        {
          "name": "John Doe",
          "company": "ACME Corp",
          "phone_number": "123-456-7890",
          "email_address": "john@acme.com"
        }
        """

        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.GenerationConfig(
                temperature=0.3,
                max_output_tokens=512
            )
        )
        response = model.generate_content([prompt, image])
        response_text = response.text

        # Extract JSON from response (remove markdown if present)
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if not json_match:
            logger.error(f"Failed to extract JSON from Gemini response: {response_text[:100]}...")
            return None, "Failed to parse LLM response as JSON"

        try:
            extracted_data = json.loads(json_match.group(0))
            # Clean extracted data
            cleaned_data = {
                "name": clean_identifier(extracted_data.get("name")),
                "company": clean_identifier(extracted_data.get("company")),
                "phone_number": clean_identifier(extracted_data.get("phone_number"), "phone_number"),
                "email_address": clean_identifier(extracted_data.get("email_address"), "email_address"),
                "gst_number": None,
                "pan_number": None,
                "address": None
            }
            logger.info(f"Gemini: Successfully extracted entities: {json.dumps(cleaned_data)}")
            return cleaned_data, None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Gemini response: {e}")
            return None, "Invalid JSON format in LLM response"

    except Exception as e:
        logger.error(f"Unexpected error in Gemini processing: {e}", exc_info=True)
        return None, f"An unexpected error occurred during image processing: {str(e)}"

# --- Text Processing Functions (Unchanged) ---
def extract_entities_with_groq(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not groq_client:
        logger.error("Groq client not initialized, cannot extract entities.")
        return None, "Groq client not initialized"
    if not text or not text.strip():
        logger.debug("No text provided for Groq entity extraction.")
        return {
            "name": None, "phone_number": None, "email_address": None, 
            "company": None, "gst_number": None, "pan_number": None, "address": None
        }, "No text provided for entity extraction"

    logger.info(f"Sending text to Groq (model: {GROQ_MODEL}) for entity extraction (length: {len(text)})")
    system_prompt = """
    You are an AI assistant specialized in accurately extracting specific fields from unstructured text, often from business cards, contact details, or user messages.
    Extract the following fields if present.
    Ensure the JSON keys in your response are *exactly* these: "name", "phone_number", "email_address", "company", "gst_number", "pan_number", "address".
    Do not invent new keys. If a field is not present or cannot be confidently extracted, use `null` for its value.

    Field definitions:
    - "name": The full name of the primary individual.
    - "phone_number": The primary phone number. Normalize by removing spaces, hyphens, parentheses, keeping only digits and a leading '+' if applicable.
    - "email_address": The primary email address, in lowercase.
    - "company": The name of the company or organization.
    - "gst_number": The GST Identification Number (GSTIN), in uppercase.
    - "pan_number": The Permanent Account Number (PAN), in uppercase.
    - "address": The full mailing address, combined into a single string if multi-line.

    Return ONLY a valid JSON object containing these fields. Prioritize accuracy. Clean up common prefixes like 'Tel:', 'Email:', 'GSTIN:', 'PAN No.:', etc.
    If multiple phone numbers or emails are present, select the primary one based on context.

    Example Input: "Meet John Doe of ACME Corp, GST 123, PAN ABC, call 9999988888 or email john@acme.com"
    Example Output:
    {
        "name": "John Doe",
        "phone_number": "9999988888",
        "email_address": "john@acme.com",
        "company": "ACME Corp",
        "gst_number": "123",
        "pan_number": "ABC",
        "address": null
    }

    Example for empty or non-extractable text:
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
            temperature=0.1,
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        result_content = chat_completion.choices[0].message.content
        logger.debug(f"Groq raw response: {result_content}")
        extracted_data_raw = json.loads(result_content)

        cleaned_data: Dict[str, Optional[str]] = {}
        expected_keys = ['name', 'phone_number', 'email_address', 'company', 'gst_number', 'pan_number', 'address']

        for key in expected_keys:
            raw_value = extracted_data_raw.get(key)
            cleaned_data[key] = clean_identifier(raw_value, key)

        logger.info(f"Groq: Successfully extracted and cleaned structured data: {json.dumps(cleaned_data)}")
        return cleaned_data, None

    except json.JSONDecodeError as e:
        logger.error(f"Groq: Failed to parse response as JSON: {e}. Response: {result_content if 'result_content' in locals() else 'N/A'}")
        return None, "Failed to parse extracted data from LLM."
    except Exception as e:
        logger.error(f"Groq: Error in entity extraction: {e}", exc_info=True)
        return None, f"LLM entity extraction failed: {str(e)}"

# --- Database Operations (Unchanged) ---
def get_all_customers() -> List[Dict[str, Any]]:
    if not supabase:
        logger.error("Supabase client not available for get_all_customers")
        return []
    try:
        response = supabase.table('customers').select(
            'id, name, phone_number, email_address, company, gst_number, pan_number, address, created_at'
        ).order('name').execute()
        return response.data if hasattr(response, 'data') else []
    except Exception as e:
        logger.error(f"Supabase: Failed to fetch all customers: {e}", exc_info=True)
        return []

async def search_customers_in_db(query_term: str, for_chat: bool = False) -> List[Dict[str, Any]]:
    if not supabase:
        logger.error("Supabase client not available for search_customers_in_db")
        return []
    
    logger.info(f"Supabase: Searching for customers with query: '{query_term}' (for_chat: {for_chat})")
    
    all_results: Dict[int, Dict[str, Any]] = {}

    def execute_and_store_sync(field_name, value_to_match, match_type="eq"):
        if not value_to_match: return
        try:
            if match_type == "ilike":
                db_response = supabase.table('customers').select('*').ilike(field_name, f'%{value_to_match}%').execute()
            else:
                db_response = supabase.table('customers').select('*').eq(field_name, value_to_match).execute()
            
            if db_response.data:
                logger.info(f"Supabase: Found {len(db_response.data)} by {field_name} {match_type} '{value_to_match}'")
                for item in db_response.data: all_results[item['id']] = item
        except Exception as e:
            logger.error(f"Supabase: Error searching by {field_name} for '{value_to_match}': {e}")

    clean_query_gst = clean_identifier(query_term, "gst_number")
    clean_query_pan = clean_identifier(query_term, "pan_number")
    clean_query_email = clean_identifier(query_term, "email_address")
    
    if clean_query_gst: execute_and_store_sync('gst_number', clean_query_gst)
    if not all_results and clean_query_pan: execute_and_store_sync('pan_number', clean_query_pan)
    if not all_results and clean_query_email: execute_and_store_sync('email_address', clean_query_email)

    clean_query_phone = clean_phone_number(query_term)
    if not all_results and clean_query_phone:
        logger.warning("PERFORMANCE_ALERT: Fetching all customers for client-side phone normalization in search_customers_in_db.")
        all_cust_for_phone = get_all_customers()
        for cust in all_cust_for_phone:
            db_phone_cleaned = clean_phone_number(cust.get('phone_number'))
            if db_phone_cleaned and db_phone_cleaned == clean_query_phone:
                all_results[cust['id']] = cust
                logger.info(f"Supabase: Found by client-side normalized phone for '{query_term}'")

    if not all_results: execute_and_store_sync('name', query_term, match_type="ilike")
    if not all_results: execute_and_store_sync('company', query_term, match_type="ilike")

    if not all_results and len(query_term) > 2:
        logger.warning("PERFORMANCE_ALERT: Fetching all customers for client-side fuzzy matching in search_customers_in_db.")
        all_customers_for_fuzzy = get_all_customers()
        fuzzy_scored_matches = []
        fields_to_fuzzy_match = ['name', 'company', 'email_address', 'phone_number', 'gst_number', 'pan_number', 'address']

        for cust in all_customers_for_fuzzy:
            max_score = 0
            matched_field = None
            for field in fields_to_fuzzy_match:
                db_value = cust.get(field) or ""
                if not db_value: continue
                threshold_key = 'phone' if field == 'phone_number' else 'email' if field == 'email_address' else field
                threshold = FUZZY_MATCH_THRESHOLDS.get(threshold_key, 75)
                score = fuzz.token_set_ratio(query_term.lower(), str(db_value).lower())
                if score > max_score and score >= threshold:
                    max_score = score
                    matched_field = field
            if max_score >= FUZZY_MATCH_THRESHOLDS.get(matched_field or 'name', 75):
                fuzzy_scored_matches.append((max_score, cust))

        if fuzzy_scored_matches:
            fuzzy_scored_matches.sort(key=lambda x: x[0], reverse=True)
            max_results_for_this_search = FUZZY_MAX_RESULTS_CHAT if for_chat else FUZZY_MAX_RESULTS_SEARCH
            for score, cust in fuzzy_scored_matches[:max_results_for_this_search]:
                if cust['id'] not in all_results:
                    all_results[cust['id']] = cust
            logger.info(f"Supabase: Added up to {min(max_results_for_this_search, len(fuzzy_scored_matches))} fuzzy matches for '{query_term}'")

    return list(all_results.values())

async def find_matching_customer_in_db(extracted_entities: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
    if not supabase or not extracted_entities:
        logger.warning("Supabase client or extracted entities missing for find_matching_customer_in_db")
        return False, None
    logger.info(f"Supabase: Finding match for extracted entities: {json.dumps(extracted_entities)}")

    unique_checks = [
        ('gst_number', clean_identifier(extracted_entities.get('gst_number'), 'gst_number')),
        ('pan_number', clean_identifier(extracted_entities.get('pan_number'), 'pan_number')),
        ('email_address', clean_identifier(extracted_entities.get('email_address'), 'email_address')),
    ]
    for field, value in unique_checks:
        if value:
            try:
                response = supabase.table('customers').select('*').eq(field, value).limit(1).execute()
                if response.data:
                    logger.info(f"Supabase: Exact match found on '{field}' for value '{value}'")
                    return True, response.data[0]
            except Exception as e:
                logger.error(f"Supabase: Error checking exact match for {field}='{value}': {e}")

    ex_phone_clean = clean_identifier(extracted_entities.get('phone_number'), 'phone_number')
    if ex_phone_clean:
        logger.warning("PERFORMANCE_ALERT: Fetching all customers for client-side phone normalization in find_matching_customer_in_db.")
        all_customers = get_all_customers()
        for customer in all_customers:
            db_phone_clean = clean_phone_number(customer.get('phone_number'))
            if db_phone_clean and db_phone_clean == ex_phone_clean:
                logger.info(f"Supabase: Exact normalized phone match found for '{ex_phone_clean}'")
                return True, customer

    logger.warning("PERFORMANCE_ALERT: Fetching all customers for client-side fuzzy matching in find_matching_customer_in_db.")
    all_customers_for_fuzzy = get_all_customers()
    if not all_customers_for_fuzzy:
        logger.info("Supabase: No customers found in database for fuzzy matching.")
        return False, None

    fuzzy_fields = ['name', 'company', 'email_address', 'phone_number', 'gst_number', 'pan_number', 'address']
    for field in fuzzy_fields:
        ex_value = clean_identifier(extracted_entities.get(field))
        if not ex_value: continue
        
        ex_value_lower = ex_value.lower()
        for customer in all_customers_for_fuzzy:
            db_value = customer.get(field) or ""
            if not db_value: continue
            threshold_key = 'phone' if field == 'phone_number' else 'email' if field == 'email_address' else field
            threshold = FUZZY_MATCH_THRESHOLDS.get(threshold_key, 75)
            score = fuzz.token_set_ratio(ex_value_lower, str(db_value).lower())
            if score >= threshold:
                logger.info(f"Supabase: Fuzzy match found on '{field}' for '{ex_value}' (score {score})")
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
        response = supabase.table('customers').insert(data_to_insert).select().execute()

        if response.data and len(response.data) > 0:
            new_customer_record = response.data[0]
            logger.info(f"Supabase: Successfully added customer ID {new_customer_record.get('id')}: {new_customer_record.get('name')}")
            return {"status": "success", "message": "Customer added successfully.", "customer_data": new_customer_record}
        else:
            error_info = getattr(response, 'error', None)
            if error_info and hasattr(error_info, 'message'):
                msg = error_info.message
                logger.error(f"Supabase: Insert failed with Supabase error: {msg}")
                if "duplicate key value violates unique constraint" in msg.lower():
                    field_match = re.search(r"constraint \"customers_(\w+)_key\"", msg)
                    field = field_match.group(1).replace('_', ' ') if field_match else "a unique field"
                    return {"status": "error", "message": f"A customer with this {field} already exists."}
                return {"status": "error", "message": f"Database error: {msg}"}
            
            logger.warning(f"Supabase: Insert command executed but no data or explicit error. Response: {response}")
            return {"status": "error", "message": "Customer addition failed. No specific error from DB."}

    except Exception as e:
        logger.error(f"Supabase: Exception during customer insertion: {e}", exc_info=True)
        err_str = str(e).lower()
        if "duplicate key value violates unique constraint" in err_str:
            field_match = re.search(r"constraint \"customers_(\w+)_key\"", err_str)
            field = field_match.group(1).replace('_', ' ') if field_match else "a unique field"
            return {"status": "error", "message": f"A customer with this {field} already exists."}
        return {"status": "error", "message": f"Failed to add customer: {str(e)}"}

# --- API Routes ---
def format_customer_details_for_chat(customer: Dict[str, Any]) -> str:
    details = []
    fields_order = ['name', 'company', 'gst_number', 'pan_number', 'email_address', 'phone_number', 'address', 'id', 'created_at']
    for field in fields_order:
        value = customer.get(field)
        if value:
            if field == 'created_at' and isinstance(value, str):
                try:
                    value_dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    value = value_dt.strftime("%Y-%m-%d %H:%M:%S %Z")
                except ValueError:
                    pass
            details.append(f"• {field.replace('_', ' ').title()}: {value}")
    return "\n".join(details) if details else "No details available."

@app.route('/api/chat', methods=['POST'])
async def handle_chat_route():
    if not request.is_json:
        return respond("Request must be JSON", "error", 415)
    
    data = request.json
    user_message = data.get('message', '').strip()

    if not user_message:
        return respond("No message provided", "error", 400)

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
            return respond("Please specify what you're looking for.", status="query_missing", code=400)
        
        logger.info(f"Chat: Search intent. Query: '{query}'")
        try:
            found_customers = await search_customers_in_db(query, for_chat=True)
        except Exception as e:
            logger.error(f"Chat: Error calling search_customers_in_db: {e}", exc_info=True)
            return respond("Error searching customers.", "error", 500)

        if not found_customers:
            return respond(f"I couldn't find customers matching '{query}'. Add them or try different details.", status="not_found")
        if len(found_customers) == 1:
            cust = found_customers[0]
            details_str = format_customer_details_for_chat(cust)
            return respond(
                f"Found customer:\n{details_str}",
                status="found_single", customer_data=cust
            )
        else:
            response_message = f"Found multiple potential matches for '{query}':\n"
            for i, c in enumerate(found_customers):
                response_message += (
                    f"{i+1}. {c.get('name', 'N/A')} "
                    f"(Company: {c.get('company', 'N/A')}, GST: {c.get('gst_number', 'N/A')})\n"
                )
            response_message += "\nPlease be more specific, or ask for details by number (e.g., 'show details for 1')."
            return respond(
                response_message,
                status="found_multiple", customer_data=found_customers
            )

    logger.info(f"Chat: No simple search, using LLM for entity extraction: '{user_message}'")
    try:
        extracted_entities, extraction_error = extract_entities_with_groq(user_message)
    except Exception as e:
        logger.error(f"Chat: Error calling extract_entities_with_groq: {e}", exc_info=True)
        return respond("Issue processing message with AI.", "error", 500)

    if extraction_error:
        return respond(f"Understanding error: {extraction_error}", status="extraction_error", code=400 if "No text" in extraction_error else 500)
    
    if not extracted_entities or not any(v for v in extracted_entities.values()):
        return respond("Couldn't extract customer details. Try rephrasing.", status="extraction_failed")

    logger.info(f"Chat: LLM extracted: {json.dumps(extracted_entities)}")
    
    try:
        match_found, existing_customer_data = await find_matching_customer_in_db(extracted_entities)
    except Exception as e:
        logger.error(f"Chat: Error calling find_matching_customer_in_db: {e}", exc_info=True)
        return respond("Error checking database.", "error", 500)

    if match_found and existing_customer_data:
        details_str = format_customer_details_for_chat(existing_customer_data)
        return respond(
            f"This looks like an existing customer:\n{details_str}\n\nIs this correct, or add new?",
            status="existing_customer_chat",
            customer_data=existing_customer_data,
            extracted_data=extracted_entities
        )
    else:
        details_summary = "\n".join([f"  • {key.replace('_', ' ').title()}: {value}" for key, value in extracted_entities.items() if value])
        return respond(
            f"Extracted details:\n{details_summary}\n\nThis seems new. Add them or use form?",
            status="new_potential_chat",
            extracted_data=extracted_entities
        )

@app.route('/api/upload-card', methods=['POST'])
async def handle_card_upload_route():
    if 'card' not in request.files:
        return respond("No 'card' file part.", "error", 400)
    
    file = request.files['card']
    if file.filename == '':
        return respond("No file selected.", "error", 400)
    if not file:
        return respond("File not provided.", "error", 400)
        
    logger.info(f"CardUpload: Received card: {file.filename}")

    try:
        extracted_entities, error = extract_entities_from_image_with_gemini(file)
        if error:
            return respond(f"Extraction failed: {error}", "extraction_failed_card",
                           code=400 if "key" in error else 500, extracted_data={})

        if not extracted_entities or not any(v for v in extracted_entities.values()):
            return respond("No structured details found.", "extraction_failed_card",
                           code=200, extracted_data=extracted_entities or {})

        logger.info(f"CardUpload: Gemini extracted: {json.dumps(extracted_entities)}")
        match_found, existing_customer_data = await find_matching_customer_in_db(extracted_entities)

        if match_found and existing_customer_data:
            logger.info(f"CardUpload: Matched existing: {existing_customer_data.get('name')}")
            return respond("Card processed. Appears to be existing customer.", "existing_customer_card",
                           matched_customer=existing_customer_data, extracted_data=extracted_entities)
        else:
            logger.info("CardUpload: No match, new potential customer.")
            return respond("Card processed. Extracted new details. Review & confirm.", "new_customer_card",
                           extracted_data=extracted_entities, matched_customer=None)
    except Exception as e:
        logger.error(f"CardUpload: Unexpected error: {e}", exc_info=True)
        return respond(f"Critical error processing card: {str(e)}", "error", 500,
                       extracted_data={})

@app.route('/api/customers', methods=['POST'])
async def add_new_customer_route():
    if not request.is_json:
        return respond("Request must be JSON", "error", 415)
    
    customer_data_from_frontend = request.json
    if not customer_data_from_frontend:
        return respond("No customer data provided.", "error", 400)

    logger.info(f"AddCustomerRoute: Data: {json.dumps(customer_data_from_frontend)}")
    try:
        add_result = await add_customer_to_db(customer_data_from_frontend)
        if add_result["status"] == "success":
            return respond(add_result["message"], status="success", code=201, customer_data=add_result.get("customer_data"))
        else:
            error_code = 409 if "already exists" in add_result["message"].lower() else 400
            return respond(add_result["message"], status="error", code=error_code)
    except Exception as e:
        logger.error(f"AddCustomerRoute: Critical error: {e}", exc_info=True)
        return respond("Server error adding customer.", "error", 500)

@app.route('/api/customers', methods=['GET'])
async def get_customers_route():
    query = request.args.get('query', '').strip()
    
    try:
        if query:
            logger.info(f"GetCustomersRoute: Searching customers with query: '{query}'")
            customers = await search_customers_in_db(query, for_chat=False)
            return respond(f"Found {len(customers)} customers matching query.", "success", customers=customers)
        else:
            logger.info("GetCustomersRoute: Fetching all customers.")
            customers = get_all_customers()
            return respond(f"Retrieved {len(customers)} customers.", "success", customers=customers)
    except Exception as e:
        logger.error(f"GetCustomersRoute: Error: {e}", exc_info=True)
        return respond("Failed to retrieve/search customers.", "error", 500)

@app.route('/health', methods=['GET'])
async def health_check_route():
    status = "ok"
    services = {
        "api_status": "running",
        "supabase_client": "initialized" if supabase else "not_initialized",
        "groq_client": "initialized" if groq_client else "not_initialized",
        "gemini_client": "configured" if genai else "not_configured"
    }
    if supabase:
        try:
            response = supabase.table('customers').select('id', count='exact').limit(0).execute()
            if hasattr(response, 'error') and response.error:
                logger.error(f"HealthCheck: Supabase query error: {response.error}")
                services["supabase_db_connection"] = f"error ({str(response.error.message)[:50]}...)"
                status = "degraded"
            elif response.count is not None:
                services["supabase_db_connection"] = f"connected (table 'customers' accessible, count={response.count})"
            else:
                logger.warning(f"HealthCheck: Supabase count query returned an unexpected response structure: {response}")
                services["supabase_db_connection"] = "query_issue_unexpected_response_structure"
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
        "__main__:asgi_app",
        host="0.0.0.0",
        port=port,
        reload=should_reload,
        log_level="debug" if should_reload else "info"
    )
