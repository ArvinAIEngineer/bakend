import os
import sqlite3
import json
import re
import requests
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS
from dotenv import load_dotenv
from groq import Groq
from fuzzywuzzy import fuzz

# --- Basic Setup ---
load_dotenv() # Load environment variables from .env file
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Keys & Config ---
NANONETS_API_KEY = os.getenv("NANONETS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
DATABASE_NAME = "customers.db"

if not NANONETS_API_KEY:
    logging.warning("NANONETS_API_KEY not found in environment variables.")
if not GROQ_API_KEY:
    logging.warning("GROQ_API_KEY not found in environment variables.")
    # Consider raising an error if Groq is essential and key is missing
    # raise ValueError("GROQ_API_KEY is required but not set.")

# --- Initialize Groq Client ---
try:
    if GROQ_API_KEY:
        groq_client = Groq(api_key=GROQ_API_KEY)
        logging.info("Groq client initialized.")
    else:
        groq_client = None
        logging.warning("Groq client not initialized due to missing API key.")
except Exception as e:
    logging.error(f"Failed to initialize Groq client: {e}")
    groq_client = None

# --- Database Setup ---
def init_db(db_name=DATABASE_NAME):
    """Initializes the database, creating the table with UNIQUE constraints."""
    conn = None
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        logging.info(f"Initializing database '{db_name}'...")

        # Create table with UNIQUE constraints directly
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS customers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                phone_number TEXT UNIQUE,
                email_address TEXT UNIQUE,
                company TEXT,
                gst_number TEXT UNIQUE,
                pan_number TEXT UNIQUE,
                address TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        logging.info("Table 'customers' checked/created.")

        # Attempt to create indexes IF NOT EXISTS.
        # NOTE: This won't fix pre-existing duplicate data if the table already
        # existed without constraints. That MUST be fixed manually beforehand.
        # This try/except mainly handles cases where the index might already
        # exist from a previous run or manual setup.
        indexes_to_create = {
            "idx_customer_phone": "phone_number",
            "idx_customer_email": "email_address",
            "idx_customer_gst": "gst_number",
            "idx_customer_pan": "pan_number"
        }
        for index_name, column_name in indexes_to_create.items():
             try:
                  create_index_sql = f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON customers ({column_name})"
                  cursor.execute(create_index_sql)
                  logging.info(f"Checked/created unique index '{index_name}' on '{column_name}'.")
             except sqlite3.IntegrityError as ie:
                 # This specifically happens if the DATA violates the constraint
                 # at the time of index creation.
                  logging.error(f"CRITICAL: Cannot create unique index '{index_name}' due to existing duplicate data in column '{column_name}'. Please clean the database manually. Error: {ie}")
                  # Optionally raise the error to stop the app, as it's a critical state
                  raise ie
             except sqlite3.OperationalError as oe:
                 # This can happen for other reasons, e.g., index already exists (though IF NOT EXISTS should handle it)
                 # or schema mismatch. Log as a warning.
                 logging.warning(f"Operational error checking/creating index '{index_name}': {oe}")


        conn.commit()
        logging.info(f"Database '{db_name}' initialization complete.")

    except sqlite3.Error as e:
        logging.error(f"Database initialization error for '{db_name}': {e}")
        # Depending on severity, you might want to exit or raise the error
        if conn:
            conn.rollback() # Rollback any partial changes
        raise e # Re-raise the error to prevent app running with bad DB state
    finally:
        if conn:
            conn.close()

# Call init_db when the application starts
# It will raise an error and stop the app if DB cleaning is needed and not done.
try:
    init_db()
except Exception as e:
    # Log the critical failure and exit if init_db fails badly
    logging.critical(f"Application cannot start due to database initialization failure: {e}")
    exit(1) # Exit the application


# --- Database Helper Functions ---
def get_db_connection(db_name=DATABASE_NAME):
    """Establishes a database connection."""
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
    return conn

def get_all_customers(db_name=DATABASE_NAME):
    conn = get_db_connection(db_name)
    cursor = conn.cursor()
    customers = []
    try:
        cursor.execute("SELECT id, name, phone_number, email_address, company, gst_number, pan_number, address FROM customers ORDER BY name COLLATE NOCASE")
        customers = [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logging.error(f"Error fetching all customers: {e}")
    finally:
        conn.close()
    return customers

def insert_customer(db_name, customer_data):
    conn = get_db_connection(db_name)
    cursor = conn.cursor()
    # Use placeholders for security
    columns = ', '.join(customer_data.keys())
    placeholders = ', '.join('?' * len(customer_data))
    sql = f'INSERT INTO customers ({columns}) VALUES ({placeholders})'
    values = tuple(customer_data.values())

    try:
        cursor.execute(sql, values)
        conn.commit()
        inserted_id = cursor.lastrowid
        logging.info(f"Inserted customer: {customer_data.get('name', 'N/A')} with ID: {inserted_id}")
        return {"status": "success", "id": inserted_id}
    except sqlite3.IntegrityError as e:
        logging.warning(f"Integrity error inserting customer (likely duplicate): {e}")
        # Determine which field caused the violation if possible (simple check)
        field = "unique field (e.g., email, phone, GST, PAN)"
        err_str = str(e).lower()
        if "phone_number" in err_str: field = "phone number"
        elif "email_address" in err_str: field = "email address"
        elif "gst_number" in err_str: field = "GST number"
        elif "pan_number" in err_str: field = "PAN number"
        return {"status": "error", "message": f"Customer with this {field} already exists."}
    except Exception as e:
        logging.error(f"Error inserting customer: {e}")
        conn.rollback() # Rollback transaction on error
        return {"status": "error", "message": f"Database error during insertion: {e}"}
    finally:
        conn.close()

def search_customer_by_name_or_gst(query, db_name=DATABASE_NAME):
    """Searches for a customer by exact GST, PAN, email, phone OR fuzzy name."""
    conn = get_db_connection(db_name)
    cursor = conn.cursor()
    results = []
    search_mode = "initial" # For logging

    try:
        # Try exact match on unique fields first
        exact_match_fields = ['gst_number', 'pan_number', 'email_address', 'phone_number']
        for field in exact_match_fields:
            cursor.execute(f"SELECT * FROM customers WHERE {field} = ?", (query,))
            customer = cursor.fetchone()
            if customer:
                results.append(dict(customer))
                search_mode = f"exact_{field}"
                logging.info(f"Found exact match on {field} for query: {query}")
                break # Stop searching if exact match found

        # If no exact match, try fuzzy name match
        if not results:
            search_mode = "fuzzy_name"
            cursor.execute("SELECT * FROM customers")
            all_customers = cursor.fetchall()
            potential_matches = []
            for cust_row in all_customers:
                customer_dict = dict(cust_row)
                name_score = fuzz.token_sort_ratio(query.lower(), (customer_dict.get("name") or "").lower())
                if name_score > 75: # Adjust threshold as needed
                    potential_matches.append((name_score, customer_dict))

            # Sort by score descending and take top N (e.g., 5)
            potential_matches.sort(key=lambda x: x[0], reverse=True)
            results = [match[1] for match in potential_matches[:5]]
            logging.info(f"Found {len(results)} potential name matches for query: {query}")

    except Exception as e:
        logging.error(f"Error searching customer (mode: {search_mode}): {e}")
    finally:
        conn.close()
    return results


def fuzzy_match_customer(extracted, customers, thresholds={
    "name": 80,
    "company": 80
}):
    """Checks for existing customer based on extracted data. Prioritizes exact unique fields."""
    if not extracted: return False, None

    best_match = None
    highest_score = 0

    # Normalize extracted data (handle potential None values)
    extracted_name = (extracted.get("name") or "").strip().lower()
    extracted_phone = re.sub(r'\D', '', extracted.get("phone_number") or "")
    extracted_email = (extracted.get("email_address") or "").strip().lower()
    extracted_company = (extracted.get("company") or "").strip().lower()
    extracted_gst = (extracted.get("gst_number") or "").strip().upper()
    extracted_pan = (extracted.get("pan_number") or "").strip().upper()

    # Fields to check for exact match first (if extracted value exists)
    exact_check_fields = []
    if extracted_phone: exact_check_fields.append(("phone_number", extracted_phone))
    if extracted_email: exact_check_fields.append(("email_address", extracted_email))
    if extracted_gst: exact_check_fields.append(("gst_number", extracted_gst))
    if extracted_pan: exact_check_fields.append(("pan_number", extracted_pan))

    for customer in customers:
        # Normalize customer data from DB
        db_phone = re.sub(r'\D', '', customer.get("phone_number") or "")
        db_email = (customer.get("email_address") or "").strip().lower()
        db_gst = (customer.get("gst_number") or "").strip().upper()
        db_pan = (customer.get("pan_number") or "").strip().upper()

        # Check exact matches first
        for field_name, extracted_value in exact_check_fields:
            db_value = locals().get(f"db_{field_name}") # Get corresponding db_ variable
            if extracted_value and db_value and extracted_value == db_value:
                 logging.info(f"Found exact match for extracted data based on {field_name}: {extracted_value}")
                 return True, customer # Return immediately on exact match

    # If no exact match found, proceed with fuzzy name/company matching
    logging.info("No exact unique field match found, proceeding with fuzzy name/company check.")
    for customer in customers:
        db_name = (customer.get("name") or "").strip().lower()
        db_company = (customer.get("company") or "").strip().lower()

        # Calculate fuzzy scores
        name_score = fuzz.token_sort_ratio(extracted_name, db_name) if extracted_name and db_name else 0
        company_score = fuzz.token_sort_ratio(extracted_company, db_company) if extracted_company and db_company else 0

        # Define match criteria (adjust logic and thresholds as needed)
        # Example: High name score OR good name and company score
        is_potential_match = (
            name_score > 90 or
            (name_score >= thresholds["name"] and company_score >= thresholds["company"])
        )

        # Use a simple average score for ranking potential matches (can be more sophisticated)
        avg_score = (name_score + company_score) / 2

        if is_potential_match and avg_score > highest_score:
             highest_score = avg_score
             best_match = customer

    if best_match:
        logging.info(f"Found potential fuzzy match for '{extracted_name}' (Score: {highest_score:.2f}) -> Matched: '{best_match.get('name')}'")
        return True, best_match
    else:
        logging.info(f"No suitable fuzzy match found for '{extracted_name}'")
        return False, None


# --- External API Helper Functions ---
def extract_text_from_image(image_file):
    if not NANONETS_API_KEY:
        logging.error("Nanonets API key not configured.")
        return None, "Nanonets API key not configured."

    nanonets_url = "https://app.nanonets.com/api/v2/OCR/FullText"
    files = {'file': (image_file.filename, image_file.read(), image_file.content_type)}
    logging.info(f"Sending file '{image_file.filename}' ({image_file.content_type}) to Nanonets...")

    try:
        response = requests.post(
            nanonets_url,
            auth=requests.auth.HTTPBasicAuth(NANONETS_API_KEY, ''),
            files=files,
            timeout=30 # Add a timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        result_json = response.json()

        # Try to extract raw text reliably
        raw_text = ""
        if result_json.get("message") == "Success" and result_json.get("results"):
            page_data = result_json["results"][0].get("page_data")
            if page_data:
                raw_text = page_data.get("raw_text", "")

        if not raw_text:
            logging.warning("Could not find 'raw_text' in expected Nanonets result structure.")
            # Add more fallback logic if needed, or return empty string
            raw_text = ""

        logging.info(f"Nanonets returned text (length: {len(raw_text)})")
        # Sanitize potentially problematic characters before returning
        safe_text = raw_text.encode('utf-8', 'replace').decode('utf-8')
        return safe_text, None

    except requests.exceptions.Timeout:
         logging.error("Nanonets API request timed out.")
         return None, "Nanonets API request timed out."
    except requests.exceptions.RequestException as e:
        logging.error(f"Nanonets API request failed: {e}")
        error_detail = ""
        try:
            error_detail = response.json().get("message", str(e))
        except: # Handle cases where response is not JSON
             error_detail = str(e)
        return None, f"Error communicating with Nanonets: {error_detail}"
    except Exception as e:
        logging.error(f"Error processing Nanonets response: {e}")
        return None, f"Unexpected error processing Nanonets result: {e}"


def extract_entities_with_groq(raw_text):
    if not groq_client:
        logging.error("Groq client not initialized. Cannot extract entities.")
        return None, "Groq client not available."
    if not raw_text or not raw_text.strip():
        logging.warning("Empty raw text received for entity extraction.")
        return {}, None # Return empty dict if no text

    logging.info(f"Sending text (length: {len(raw_text)}) to Groq for entity extraction...")
    try:
        # Llama 3.1 Instruct recommended for JSON mode as of July 2024
        model_to_use = "llama-3.3-70b-versatile"
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": (
                    "You are an AI assistant specialized in accurately extracting specific fields from unstructured text, often from business cards or contact details. "
                    "Extract the following fields if present: 'name', 'phone_number', 'email_address', 'company', 'gst_number', 'pan_number', 'address'. "
                    "Return ONLY a valid JSON object containing the extracted fields. Use null for missing fields or if a field cannot be confidently extracted. "
                    "Prioritize accuracy. Clean up common prefixes like 'Tel:', 'Email:', 'GSTIN:', etc. Consolidate multi-line addresses into a single string."
                    "Format phone numbers consistently if possible (e.g., +91 XXXXXXXXXX or XXX-XXX-XXXX), but preserve original if unsure."
                    """
                    Example Input:
                    John K. Doe
                    Chief Executive Officer
                    Acme Innovations Ltd.
                    123 Innovation Drive, Suite 400
                    Tech City, ST 12345
                    Tel: +91 98765 43210
                    Mobile: 888-555-1212
                    Email: john.doe@acmeinnovate.com
                    GSTIN: 27ABCDE1234F1Z5

                    Example Output:
                    {
                        "name": "John K. Doe",
                        "phone_number": "+919876543210",
                        "email_address": "john.doe@acmeinnovate.com",
                        "company": "Acme Innovations Ltd.",
                        "gst_number": "27ABCDE1234F1Z5",
                        "pan_number": null,
                        "address": "123 Innovation Drive, Suite 400, Tech City, ST 12345"
                    }
                    """
                )},
                {"role": "user", "content": raw_text}
            ],
            model=model_to_use,
            temperature=0.1, # Very low temperature for deterministic extraction
            max_tokens=512,
            response_format={"type": "json_object"}, # Enforce JSON output
        )
        result_content = chat_completion.choices[0].message.content
        logging.info(f"Groq extraction successful (Model: {model_to_use}).")

        # Parse the JSON string
        extracted_data = json.loads(result_content)

        # Basic validation/cleaning
        cleaned_data = {}
        valid_keys = ['name', 'phone_number', 'email_address', 'company', 'gst_number', 'pan_number', 'address']
        for key in valid_keys:
            value = extracted_data.get(key)
            # Ensure value is string or None/null, trim whitespace
            cleaned_data[key] = str(value).strip() if value is not None and isinstance(value, (str, int, float)) else None
            if key == 'phone_number' and cleaned_data[key]:
                 # Optional: Further phone number cleaning
                 cleaned_data[key] = re.sub(r'[^\d+]', '', cleaned_data[key]) # Keep digits and '+'
            elif key in ['gst_number', 'pan_number'] and cleaned_data[key]:
                 cleaned_data[key] = cleaned_data[key].upper() # Ensure case consistency

        logging.info(f"Cleaned extracted data: {cleaned_data}")
        return cleaned_data, None

    except json.JSONDecodeError as e:
        logging.error(f"Groq response was not valid JSON: {e}\nResponse received: {result_content}")
        return None, "Failed to parse extracted data as JSON."
    except Exception as e:
        # Catch potential Groq API errors (e.g., rate limits, invalid request)
        logging.error(f"Groq API call failed: {e}")
        return None, f"Error communicating with Groq: {e}"


# --- API Endpoints ---

@app.route('/api/chat', methods=['POST'])
def handle_chat():
    """Handles general chat messages and directs logic."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415 # Unsupported Media Type
    data = request.json
    message = data.get('message', '').strip() # Trim whitespace
    logging.info(f"Received chat message: '{message}'")

    if not message:
        return jsonify({"reply": "Sorry, I didn't receive a message."}), 400

    lowercase_message = message.lower()

    # Simple Greeting
    if lowercase_message == 'hi':
        return jsonify({
            "reply": "Hello! How can I help? Try:\n• 'find customer [name/GST/PAN/email/phone]'\n• 'check details for [name] with email [email]...' (provide some details)\n• Or use the form to add a new customer / upload a card.",
             "status": "greeting"
        })

    # Find Customer Intent (more robust regex)
    # Allows "find customer", "search for customer", "lookup customer", etc.
    find_match = re.match(r"(?:find|search|lookup) customer (.+)", lowercase_message)
    if find_match:
        query = find_match.group(1).strip()
        if not query:
             return jsonify({"reply": "Please specify what customer name or ID to find.", "status": "query_missing"})

        logging.info(f"Searching for customer with query: '{query}'")
        results = search_customer_by_name_or_gst(query)

        if results:
            if len(results) == 1:
                 cust = results[0]
                 reply = f"Found customer:\n" \
                         f"  Name: {cust.get('name', 'N/A')}\n" \
                         f"  Company: {cust.get('company', 'N/A')}\n" \
                         f"  GST: {cust.get('gst_number', 'N/A')}\n" \
                         f"  PAN: {cust.get('pan_number', 'N/A')}\n" \
                         f"  Email: {cust.get('email_address', 'N/A')}\n" \
                         f"  Phone: {cust.get('phone_number', 'N/A')}"
                 return jsonify({"reply": reply, "customer_data": cust, "status": "found_single"})
            else:
                 reply = f"Found multiple potential matches for '{query}'. Please be more specific:\n" + "\n".join(
                     [f"- {c.get('name')} (GST: {c.get('gst_number', 'N/A')}, Email: {c.get('email_address', 'N/A')})" for c in results]
                 )
                 return jsonify({"reply": reply, "customer_data": results, "status": "found_multiple"})
        else:
             reply = f"Sorry, I couldn't find a customer matching '{query}'. You can try providing more details or use the form to add a new customer."
             return jsonify({"reply": reply, "status": "not_found"})

    # Check/Extract Details Intent (Manual Entry Simulation)
    # Catches "check details for...", "add customer...", etc.
    check_match = re.match(r"(?:check|verify|add|enter) details? (?:for|customer)?(.+)", lowercase_message, re.IGNORECASE)
    if check_match:
        details_text = check_match.group(1).strip()
        if not details_text:
             return jsonify({"reply": "Please provide the customer details you want to check.", "status": "details_missing"})

        logging.info(f"Attempting to extract entities from manual input: '{details_text[:60]}...'")
        extracted_data, error = extract_entities_with_groq(details_text)

        if error:
            logging.error(f"Entity extraction failed for manual input: {error}")
            return jsonify({"reply": f"Error extracting details: {error}"}), 500
        if not extracted_data or not any(extracted_data.values()): # Check if extraction yielded anything
             logging.warning(f"No entities extracted from manual input: '{details_text}'")
             return jsonify({
                 "reply": "Sorry, I couldn't extract structured details from that. Could you try formatting it clearly? (e.g., Name: John Doe, Email: john@example.com, GST: ...)",
                 "status": "extraction_failed"
             })

        logging.info(f"Extracted data from manual input: {extracted_data}")
        all_customers = get_all_customers()
        is_match, matched_customer = fuzzy_match_customer(extracted_data, all_customers)

        if is_match and matched_customer:
            reply = f"Based on the details, this might be an existing customer:\n" \
                    f"  Name: {matched_customer.get('name', 'N/A')}\n" \
                    f"  Company: {matched_customer.get('company', 'N/A')}\n" \
                    f"  GST: {matched_customer.get('gst_number', 'N/A')}"
            # You could add a prompt here asking if this is the correct one or if they still want to add
            return jsonify({"reply": reply, "status": "existing_match", "customer_data": matched_customer, "extracted_data": extracted_data})
        else:
            reply = "Based on the details, this looks like a new customer.\n\nExtracted Info:\n"
            extracted_info_str = ""
            for key, value in extracted_data.items():
                if value: # Only show non-empty extracted fields
                    extracted_info_str += f"  - {key.replace('_', ' ').title()}: {value}\n"

            if not extracted_info_str:
                 reply += "  (No specific details extracted)\n"
            else:
                 reply += extracted_info_str

            reply += "\nIf this looks correct, you can use the 'Add Customer' form (I can pre-fill it for you). Or, upload a visiting card for potentially more accurate extraction."
            # We return the extracted data so the frontend can pre-fill the form
            return jsonify({"reply": reply, "status": "new_potential", "extracted_data": extracted_data})

    # Fallback response for unrecognized input
    logging.warning(f"Unrecognized chat message: '{message}'")
    return jsonify({
        "reply": "Sorry, I didn't understand that. Please try 'find customer [query]' or 'check details for [details...]', or use the form to add a customer.",
        "status": "unrecognized"
    })


@app.route('/api/upload-card', methods=['POST'])
def upload_card():
    """Handles visiting card upload, extracts text, entities, and checks DB."""
    if 'card' not in request.files:
        logging.warning("Upload request received without 'card' file part.")
        return jsonify({"error": "No 'card' file part in the request"}), 400

    file = request.files['card']
    if not file or file.filename == '':
        logging.warning("Upload request received with empty file.")
        return jsonify({"error": "No file selected or file is empty"}), 400

    # Optional: Add file type/size validation here if not done in frontend
    # allowed_extensions = {'png', 'jpg', 'jpeg', 'pdf'}
    # if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
    #     return jsonify({"error": "Invalid file type"}), 400

    logging.info(f"Processing uploaded card: {file.filename} ({file.content_type})")

    # 1. Extract Text
    raw_text, error_text = extract_text_from_image(file)
    if error_text:
        return jsonify({"error": f"Text extraction failed: {error_text}"}), 500
    if not raw_text:
        logging.warning(f"No text could be extracted from file: {file.filename}")
        return jsonify({"error": "Could not extract any text from the image."}), 400

    # 2. Extract Entities
    extracted_data, error_entities = extract_entities_with_groq(raw_text)
    if error_entities:
        return jsonify({"error": f"Entity extraction failed: {error_entities}"}), 500
    if not extracted_data:
         logging.warning(f"No structured entities extracted from text of file: {file.filename}")
         # Still return raw text and allow manual entry
         return jsonify({
             "error": "Could not extract structured data, but text was read.",
             "raw_text": raw_text,
             "extracted_data": {},
             "match_status": "extraction_failed",
             "matched_customer": None
         }), 400 # Or 200 if you want the frontend to handle this state

    # 3. Check Database
    all_customers = get_all_customers()
    is_match, matched_customer = fuzzy_match_customer(extracted_data, all_customers)

    response_data = {
        "extracted_data": extracted_data,
        "raw_text": raw_text, # Optionally return raw text
        "match_status": "existing_match" if is_match else "new_potential",
        "matched_customer": matched_customer if is_match else None
    }

    logging.info(f"Card processed for '{file.filename}'. Match status: {response_data['match_status']}")
    return jsonify(response_data), 200


@app.route('/api/customers', methods=['POST'])
def add_customer_route():
    """Adds a new customer to the database via POST request."""
    if not request.is_json:
         return jsonify({"status": "error", "message": "Request must be JSON"}), 415

    customer_data = request.json
    logging.info(f"Received request to add customer: {customer_data.get('name', 'N/A')}")

    # --- Server-side Validation ---
    required_fields = ['name', 'gst_number', 'pan_number', 'address'] # Adjust as per your business logic
    missing_fields = [f for f in required_fields if not customer_data.get(f)]
    if missing_fields:
         logging.warning(f"Add customer request missing fields: {', '.join(missing_fields)}")
         return jsonify({"status": "error", "message": f"Missing required fields: {', '.join(missing_fields)}"}), 400

    # Basic format validation (examples)
    if customer_data.get('email_address') and '@' not in customer_data['email_address']:
         return jsonify({"status": "error", "message": "Invalid email format"}), 400
    # Add regex validation for GST/PAN if needed

    # Filter only valid columns for insertion to prevent injection/errors
    valid_columns = ['name', 'phone_number', 'email_address', 'company', 'gst_number', 'pan_number', 'address']
    data_to_insert = {
        k: str(v).strip() if v is not None else None
        for k, v in customer_data.items() if k in valid_columns
    }
    # Ensure required fields aren't accidentally filtered out if they were empty strings initially
    for req_field in required_fields:
        if req_field not in data_to_insert and customer_data.get(req_field) == "":
            data_to_insert[req_field] = "" # Or handle as error if empty not allowed

    # Remove keys with None values if your DB schema doesn't handle them well or they aren't nullable
    # data_to_insert = {k: v for k, v in data_to_insert.items() if v is not None}

    if not data_to_insert.get('name'): # Check at least name is present after filtering
         return jsonify({"status": "error", "message": "Valid name is required."}), 400

    logging.info(f"Attempting to insert customer data: {data_to_insert}")
    result = insert_customer(DATABASE_NAME, data_to_insert)

    if result["status"] == "success":
        return jsonify(result), 201 # 201 Created
    else:
        # Return 409 Conflict for duplicates, 400 for validation, 500 for other DB errors
        status_code = 409 if "already exists" in result.get("message", "") else 500
        # Override with 400 if it was a validation error we caught earlier but somehow got here
        # (This part of the logic might be redundant if validation is strict above)
        # status_code = 400 if "validation" in result.get("message", "").lower() else status_code
        logging.error(f"Failed to add customer. Result: {result}")
        return jsonify(result), status_code


# --- Simple Health Check Endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    # Optionally add checks for DB connection, external services etc.
    return jsonify({"status": "ok"}), 200

# --- Run Flask App ---
if __name__ == '__main__':
    # Use environment variable for port, default to 5000
    port = int(os.environ.get("PORT", 5200))
    # Use host='0.0.0.0' to be accessible externally (e.g., in Docker)
    # Use debug=True only for development, set to False for production
    is_debug = os.environ.get("FLASK_DEBUG", "True").lower() == "true"
    logging.info(f"Starting Flask server on port {port} with debug={is_debug}")
    app.run(debug=is_debug, host='0.0.0.0', port=port)
