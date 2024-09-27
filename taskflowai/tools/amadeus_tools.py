import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import requests

class AmadeusTools:
    @staticmethod
    def _get_access_token():
        """Get Amadeus API access token."""
        load_dotenv()
        api_key = os.getenv("AMADEUS_API_KEY")
        api_secret = os.getenv("AMADEUS_API_SECRET")
        
        if not api_key or not api_secret:
            raise ValueError("AMADEUS_API_KEY and AMADEUS_API_SECRET must be set in .env file")
        
        token_url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        data = {
            "grant_type": "client_credentials",
            "client_id": api_key,
            "client_secret": api_secret
        }
        
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        return response.json()["access_token"]

    @staticmethod
    def search_flights(
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1,
        children: int = 0,
        infants: int = 0,
        travel_class: Optional[str] = None,
        non_stop: bool = False,
        currency: str = "USD",
        max_price: Optional[int] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Search for flight offers using Amadeus API.

        Args:
            origin (str): Origin airport (e.g., "YYZ")
            destination (str): Destination airport (e.g., "CDG")
            departure_date (str): Departure date in YYYY-MM-DD format
            return_date (Optional[str]): Return date in YYYY-MM-DD format for round trips
            adults (int): Number of adult travelers
            children (int): Number of child travelers
            infants (int): Number of infant travelers
            travel_class (Optional[str]): Preferred travel class (ECONOMY, PREMIUM_ECONOMY, BUSINESS, FIRST)
            non_stop (bool): If True, search for non-stop flights only
            currency (str): Currency code for pricing (default: USD)
            max_price (Optional[int]): Maximum price per traveler
            max_results (int): Maximum number of results to return (default: 10)

        Returns:
            Dict[str, Any]: Flight search results
        """
        access_token = AmadeusTools._get_access_token()
        
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "children": children,
            "infants": infants,
            "currencyCode": currency,
            "max": max_results
        }
        
        if return_date:
            params["returnDate"] = return_date
        
        if travel_class:
            params["travelClass"] = travel_class
        
        if non_stop:
            params["nonStop"] = "true"
        
        if max_price:
            params["maxPrice"] = max_price
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            error_message = f"Error searching for flight offers: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                error_message += f"\nResponse status code: {e.response.status_code}"
                error_message += f"\nResponse content: {e.response.text}"
            return error_message

    @staticmethod
    def get_cheapest_date(
        origin: str,
        destination: str,
        departure_date: str,
        return_date: Optional[str] = None,
        adults: int = 1
    ) -> Dict[str, Any]:
        """
        Find the cheapest travel dates for a given route using the Flight Offers Search API.

        Args:
            origin (str): IATA code of the origin airport.
            destination (str): IATA code of the destination airport.
            departure_date (str): Departure date in YYYY-MM-DD format.
            return_date (Optional[str]): Return date in YYYY-MM-DD format for round trips. Defaults to None.
            adults (int): Number of adult travelers. Defaults to 1.

        Returns:
            Dict[str, Any]: A dictionary containing the cheapest flight offer information, including:
                - price: The total price of the cheapest offer.
                - departureDate: The departure date of the cheapest offer.
                - returnDate: The return date of the cheapest offer (if applicable).
                - airline: The airline code of the cheapest offer.
                - additional details about the flight itinerary.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.

        Note:
            This method uses the Amadeus Flight Offers Search API to find the cheapest flight
            for the given route and dates. It returns only one result (the cheapest offer).
        """
        access_token = AmadeusTools._get_access_token()
        
        url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": departure_date,
            "adults": adults,
            "max": 1  # We only need the cheapest offer
        }
        
        if return_date:
            params["returnDate"] = return_date
        
        response = requests.get(url, headers=headers, params=params)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response content: {response.text}")
            raise
        return response.json()

    @staticmethod
    def get_flight_inspiration(
        origin: str,
        max_price: Optional[int] = None,
        currency: str = "EUR"
    ) -> Dict[str, Any]:
        """
        Get flight inspiration using the Flight Inspiration Search API.

        This method uses the Amadeus Flight Inspiration Search API to find travel destinations
        based on the origin city and optional price constraints.

        Args:
            origin (str): IATA code of the origin city.
            max_price (Optional[int], optional): Maximum price for the flights. Defaults to None.
            currency (str, optional): Currency for the price. Defaults to "EUR".

        Returns:
            Dict[str, Any]: A dictionary containing flight inspiration results, including:
                - data: A list of dictionaries, each representing a destination with details such as:
                    - type: The type of the result (usually "flight-destination").
                    - origin: The IATA code of the origin city.
                    - destination: The IATA code of the destination city.
                    - departureDate: The suggested departure date.
                    - returnDate: The suggested return date.
                    - price: The price information for the trip.

        Raises:
            requests.exceptions.HTTPError: If the API request fails.
            ValueError: If the required environment variables are not set.

        Note:
            This method requires valid Amadeus API credentials to be set in the environment variables.
        """
        access_token = AmadeusTools._get_access_token()
        
        url = "https://test.api.amadeus.com/v1/shopping/flight-destinations"
        
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "origin": origin,
            "currency": currency
        }
        
        if max_price:
            params["maxPrice"] = max_price
        
        response = requests.get(url, headers=headers, params=params)
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            print(f"Response content: {response.text}")
            raise
        return response.json()
