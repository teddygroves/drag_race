from __future__ import print_function
import pickle
import os.path
import pandas as pd
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SHEET_ID = '1Sotvl3o7J_ckKUg5sRiZTqNQn3hPqhepBSeOpMTK15Q'
SHEET_NAMES = [
     'all_episodes', 'all_rankings', 'all_contestants', 'all_social_media'
]


def get_google_sheet(spreadsheet_id, range_name):
    """Shows basic usage of the Sheets API"""
    creds = None
    # The file token.pickle stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                 'google_sheets_credentials.json', SCOPES)
            creds = flow.run_local_server()
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    service = build('sheets', 'v4', credentials=creds)

    # Call the Sheets API
    sheet = service.spreadsheets()
    return sheet.values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()


def get_df_from_google_sheet(gsheet):
    """ Converts Google sheet data to a Pandas DataFrame.
    """
    header = gsheet.get('values', [])[0]   # Assumes first line is header!
    values = gsheet.get('values', [])[1:]  # Everything else is data.
    if not values:
        print('No data found.')
    else:
        return pd.DataFrame(values, columns=header)


def main():
    for sheet_name in SHEET_NAMES:
        sheet = get_google_sheet(SHEET_ID, sheet_name)
        dataframe = get_df_from_google_sheet(sheet)
        dataframe.to_csv(f'data/{sheet_name}.csv')


if __name__ == '__main__':
    main()
