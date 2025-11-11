import requests
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta  # <-- Import datetime
import time  # <-- Import time for a small delay

# --- 1. SETUP & CONSTANTS ---
load_dotenv()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

AMADEUS_CLIENT_ID = os.getenv("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.getenv("AMADEUS_CLIENT_SECRET")
AMADEUS_BASE_URL = "https://test.api.amadeus.com"

# --- 2. AUTHENTICATION FUNCTION ---
def get_amadeus_token():
    print("Fetching Amadeus token...")
    auth_url = f"{AMADEUS_BASE_URL}/v1/security/oauth2/token"
    auth_data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET,
    }
    try:
        response = requests.post(auth_url, data=auth_data)
        response.raise_for_status()
        token = response.json()['access_token']
        print("Token fetched successfully.")
        return token
    except requests.exceptions.HTTPError as err:
        print(f"Error fetching token: {err.response.json()}")
        return None

# --- 3. PARSING FUNCTION ---
def parse_flight_data(amadeus_response, search_date):
    """
    Parses the Amadeus JSON response and adds the search date.
    """
    flight_offers = amadeus_response.get('data', [])
    parsed_list = []

    for offer in flight_offers:
        try:
            itinerary = offer['itineraries'][0]
            segment = itinerary['segments'][0]
            fare_detail = offer['travelerPricings'][0]['fareDetailsBySegment'][0]
            
            parsed_offer = {
                "Search Date": search_date,  # <-- Added the date we searched for
                "Total Price": offer['price']['total'],
                "Last Day to Book": offer['lastTicketingDate'],
                "Airline": segment['carrierCode'],
                "Aircraft": segment['aircraft']['code'],
                "Departure Airport": segment['departure']['iataCode'],
                "Departure Time": segment['departure']['at'],
                "Arrival Airport": segment['arrival']['iataCode'],
                "Arrival Time": segment['arrival']['at'],
                "Cabin": fare_detail['cabin']
            }
            parsed_list.append(parsed_offer)
        except (KeyError, IndexError) as e:
            print(f"Skipping malformed offer (ID: {offer.get('id')}): {e}")

    return parsed_list

# --- 4. MAIN EXECUTION (MODIFIED) ---
if __name__ == "__main__":
    
    # --- Define your search criteria (date will be set in the loop) ---
    search_params = {
        "originLocationCode": "DEL",
        "destinationLocationCode": "MAA",
        "adults": "1",
        "currencyCode": "INR",
        "includedAirlineCodes": "AI",
        "max": 10
    }
    
    # This will store results from all 10 days
    all_flight_results = []
    
    # 1. Get Token (only once)
    token = get_amadeus_token()

    if token:
        headers = {"Authorization": f"Bearer {token}"}
        search_url = f"{AMADEUS_BASE_URL}/v2/shopping/flight-offers"
        
        # We start from tomorrow's date
        start_date = datetime.now() + timedelta(days=1)

        # 2. Loop for the next 10 days
        for i in range(10):
            # Calculate the date for this loop iteration
            current_search_date = start_date + timedelta(days=i)
            date_str = current_search_date.strftime('%Y-%m-%d')
            
            # Add the specific date to our search parameters
            search_params['departureDate'] = date_str
            
            print(f"\n--- Searching for flights on: {date_str} ---")

            try:
                # 3. Search for Flights for this specific day
                response = requests.get(search_url, headers=headers, params=search_params)
                response.raise_for_status()
                raw_data = response.json()
                
                # 4. Parse Data
                simplified_data = parse_flight_data(raw_data, date_str)
                
                if simplified_data:
                    print(f"Found {len(simplified_data)} flights.")
                    # Add this day's results to our master list
                    all_flight_results.extend(simplified_data)
                else:
                    print("No flights found for this date in the test data.")
                
                # Be polite to the API - add a tiny delay
                time.sleep(0.1) 

            except requests.exceptions.HTTPError as err:
                print(f"API Error for date {date_str}: {err.response.json()}")
            except Exception as e:
                print(f"An error occurred on {date_str}: {e}")

        # 5. --- After the loop, create the final DataFrame and CSV ---
        if not all_flight_results:
            print("\nNo flights found for any of the 10 days.")
        else:
            df = pd.DataFrame(all_flight_results)
            
            output_filename = "flight_results_10_days.csv"
            df.to_csv(output_filename, index=False, encoding='utf-8')
            
            print("\n--- Combined Flight Search Results (All 10 Days) ---")
            print(df)
            print(f"\nSuccessfully saved all data to {output_filename}")