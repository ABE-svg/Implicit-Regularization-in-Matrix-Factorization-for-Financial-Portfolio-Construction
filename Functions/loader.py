import yfinance as yf
import pandas   as pd
from   datetime import datetime
import holidays 
import pandas_market_calendars as mcal

class MarketData:
    def __init__(self,tickers, start_date="2000-01-01", end_date=None):
        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date or datetime.now().strftime('%Y-%m-%d')

    def download_data(self):
        """
        Download adjusted closing prices from Yahoo Finance for a list of tickers.
        Clean the data (NA, consistency) and return a clean DataFrame.
        """
        data = yf.download(self.tickers, start=self.start_date, end=self.end_date, auto_adjust=True)

        prices = data["Close"]                  # Only closing prices are retained.

        #prices = prices.dropna(how="all")       # Remove completely empty lines
        #prices = prices.dropna()                # deletes any incomplete dates
    
        return prices
    

    def check_date_continuity(self, market='NYSE', country='US'):
        """
        Checks the continuity of dates in a financial DataFrame.
        Provides detailed information on missing dates:
        - weekend
        - public holiday
        - exceptional market closure
        Displays a final message indicating whether the missing dates are consistent or not.
        """
        df = self.download_data()
        idx = df.index

        print("\n Date verification ")

        # Duplicate checking
        if idx.duplicated().sum() > 0:
            print(f"{idx.duplicated().sum()} duplicate dates")
        else:
            print("No duplicate dates")

        # Verification of NAs by line
        missing_by_row = df.isna().sum(axis=1)
        print(f"Number of lines with internal NAs: {sum(missing_by_row > 0)}")

        # Official scholarship calendar
        cal = mcal.get_calendar(market.upper())
        schedule = cal.schedule(start_date=idx.min(), end_date=idx.max())
        market_open_dates = schedule.index
 
        # Missing dates according to the trading calendar
        missing = market_open_dates.difference(idx)

        inconsistent_count = 0  # compteur de fermetures inattendues

        if len(missing) == 0:
            print("No missing dates on market open days")
        else:
            print(f"{len(missing)} missing dates on market open days:")

        # Holiday calendar
        if country.upper() == 'US':
            country_holidays = holidays.US()
        elif country.upper() == 'FR':
            country_holidays = holidays.France()
        else:
            country_holidays = holidays.CountryHoliday(country)

        for d in missing:
            reason = ""
            if d.weekday() >= 5:  # Saturday or Sunday
                reason = "Weekend (market closed)"
            elif d in country_holidays:
                reason = f"Public holiday: {country_holidays.get(d)}"
            else:
                reason = "Market closed (unexpected closure)"
                inconsistent_count += 1

            print(f" - {d.date()} - {reason}")

        # Final message
        if inconsistent_count == 0:
            print("\nAll missing dates are consistent with the trading calendar")
        else:
            print(f"\nWarning: {inconsistent_count} missing dates are inconsistent with the trading calendar")

        print("--- End of verification ---\n")

