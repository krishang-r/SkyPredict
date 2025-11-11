# data_collector.py

import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta  # Import datetime
import time  # Import time for a small delay

# --- 1. SETUP & CONSTANTS ---
load_dotenv()  # Load variables from .env file

# Set pandas to display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
AMADEUS_BASE_URL = "https://test.api.amadeus.com"

# --- 2. AUTHENTICATION FUNCTION ---
def get_amadeus_token():
    """
    Fetches the Amadeus API access token.
    """
    print("Fetching Amadeus token...")
    auth_url = f"{AMADEUS_BASE_URL}/v1/security/oauth2/token"
    
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET,
    }
    
    try:
        response = requests.post(auth_url, data=auth_data)
        response.raise_for_status()  # Raise error for bad responses (4xx or 5xx)
        token = response.json()['access_token']
        print("Token fetched successfully.")
        return token
    except requests.exceptions.HTTPError as err:
        print(f"Error fetching token: {err.response.json()}")
        return None

# --- 3. PARSING FUNCTION ---
def parse_flight_data(amadeus_response, days_until_departure):
    """
    Parses the Amadeus JSON response to extract specific fields.
    """
    flight_offers = amadeus_response.get('data', [])
    parsed_list = []

    for offer in flight_offers:
        try:
            # Assumes one-way, direct flight (one itinerary, one segment)
            itinerary = offer['itineraries'][0]
            segment = itinerary['segments'][0]
            fare_detail = offer['travelerPricings'][0]['fareDetailsBySegment'][0]
            
            parsed_offer = {
                # This new feature is CRITICAL for the model
                "Days Until Departure": days_until_departure, 
                "Total Price": float(offer['price']['total']), # Convert to float for ML
                "Airline": segment['carrierCode'],
                "Aircraft": segment['aircraft']['code'],
                "Departure Airport": segment['departure']['iataCode'],
                "Arrival Airport": segment['arrival']['iataCode'],
                "Departure Time": segment['departure']['at'],
                "Cabin": fare_detail['cabin']
            }
            parsed_list.append(parsed_offer)
        except (KeyError, IndexError, TypeError):
            pass # Silently skip malformed offers

    return parsed_list

# --- 4. MAIN EXECUTION (MODIFIED TO INCLUDE RETURN ROUTES) ---
if __name__ == "__main__":
    
    # --- YOUR NEW DATASET PARAMETERS ---
    # This is our base list of one-way routes
    ONE_WAY_ROUTES = [
        ("DEL", "MAA"),
        ("DEL", "BOM"),
        ("BOM", "GOI"),
        ("BLR", "CCU"),
        ("DEL", "BLR"),
        ("BOM", "DXB") # (DXB -> BOM will be added)
    ]
    
    # --- NEW: Create a full list of all routes (one-way + return) ---
    all_routes_to_search = []
    for origin, dest in ONE_WAY_ROUTES:
        all_routes_to_search.append((origin, dest)) # Add original
        all_routes_to_search.append((dest, origin)) # Add return
    
    print(f"Set to search {len(all_routes_to_search)} total routes (one-way and return).")
    
    # Search for all major airlines at once
    AIRLINES_TO_SEARCH = "AI,6E,UK,SG,QP" 
    DAYS_TO_SEARCH = 20 # Let's search 20 days into the future
    
    all_flight_results = []
    token = get_amadeus_token()

    if token:
        headers = {"Authorization": f"Bearer {token}"}
        search_url = f"{AMADEUS_BASE_URL}/v2/shopping/flight-offers"
        start_date = datetime.now() + timedelta(days=1)

        # Loop 1: Over each route (now includes return routes)
        for origin, dest in all_routes_to_search:
            
            # Loop 2: Over each of the next N days
            for i in range(DAYS_TO_SEARCH):
                days_until_departure = i + 1
                current_search_date = start_date + timedelta(days=i)
                date_str = current_search_date.strftime('%Y-%m-%d')
                
                search_params = {
                    "originLocationCode": origin,
                    "destinationLocationCode": dest,
                    "departureDate": date_str,
                    "adults": "1",
                    "currencyCode": "INR",
                    "includedAirlineCodes": AIRLINES_TO_SEARCH,
                    "max": 10 # Get a few results for each
                }
                
                print(f"Searching: {origin}->{dest} on {date_str}...")

                try:
                    response = requests.get(search_url, headers=headers, params=search_params)
                    response.raise_for_status()
                    raw_data = response.json()
                    
                    simplified_data = parse_flight_data(raw_data, days_until_departure)
                    
                    if simplified_data:
                        all_flight_results.extend(simplified_data)
                
                except requests.exceptions.HTTPError as err:
                    # Don't stop the script, just log the error
                    # Use .get() for safe dictionary access to avoid errors
                    error_detail = err.response.json().get('errors', [{}])[0].get('detail', 'Unknown error')
                    print(f"  API Error (skipping): {error_detail}")
                
                time.sleep(0.2) # Be polite to the API

        if all_flight_results:
            df = pd.DataFrame(all_flight_results)
            output_filename = "flight_dataset.csv"
            df.to_csv(output_filename, index=False, encoding='utf-8')
            print(f"\nSUCCESS: Saved {len(all_flight_results)} flight offers to {output_filename}")
        else:
            print("\nNo data collected. Check API or parameters.")