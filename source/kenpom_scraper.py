import os
import pandas as pd
import numpy as np
import re
import requests

from bs4 import BeautifulSoup

FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(FILEPATH, '../data/'))

base_urls = {
    2011: 'https://web.archive.org/web/20110311233233/http://www.kenpom.com/',
    2012: 'https://web.archive.org/web/20120311165019/http://kenpom.com/',
    2013: 'https://web.archive.org/web/20130318221134/http://kenpom.com/',
    2014: 'https://web.archive.org/web/20140318100454/http://kenpom.com/',
    2015: 'https://web.archive.org/web/20150316212936/http://kenpom.com/',
    2016: 'https://web.archive.org/web/20160314134726/http://kenpom.com/',
    2017: 'https://web.archive.org/web/20170312131016/http://kenpom.com/',
    2018: 'https://web.archive.org/web/20180311122559/https://kenpom.com/',
    2019: 'https://web.archive.org/web/20190317211809/https://kenpom.com/',
    2021: 'https://web.archive.org/web/20210317211809/https://kenpom.com/',
    2022: 'https://web.archive.org/web/20220315211809/https://kenpom.com/',
    2023: 'https://web.archive.org/web/20230313211809/https://kenpom.com/',
    2024: 'https://web.archive.org/web/20240317211809/https://kenpom.com/',
}


def scrape_single_year(year):
    """
    Scrapes raw data from a kenpom archive page and returns it as
    a dataframe.
    """
    url = base_urls[year]
    print(f'Scraping: {url}')

    page = requests.get(base_urls[year])
    soup = BeautifulSoup(page.text)
    table_full = soup.find_all('table', {'id': 'ratings-table'})

    thead = table_full[0].find_all('thead')
    table = table_full[0]

    for weird in thead:
        table = str(table).replace(str(weird), '')

    df = pd.read_html(table)[0]
    df['year'] = year

    return df


def get_kenpom_data(years):
    '''
    Get pre-tourney KenPom data for all years in
    the specified years list and return as a cleaned
    up DataFrame.
    '''
    archive_list = []
    for year in years:

        archive = scrape_single_year(year)
        archive_list.append(archive)

    df = pd.concat(archive_list, axis=0)

    df.columns = ['Rank', 'Team', 'Conference', 'W-L', 'AdjEM',
                  'AdjustO', 'AdjustO Rank', 'AdjustD', 'AdjustD Rank',
                  'AdjustT', 'AdjustT Rank', 'Luck', 'Luck Rank',
                  'SOS Pyth', 'SOS Pyth Rank', 'SOS OppO', 'SOS OppO Rank',
                  'SOS OppD', 'SOS OppD Rank', 'NCSOS Pyth', 'NCSOS Pyth Rank',
                  'Year']
    col_mapping = {
        'Year': 'Season',
        'Team': 'TeamName',
        'AdjustO': 'kenpom_adj_o',
        'AdjustO Rank': 'kenpom_adj_o_rank',
        'AdjustD': 'kenpom_adj_d',
        'AdjustD Rank': 'kenpom_adj_d_rank',
        'Rank': 'kenpom_rank',
        'Luck': 'kenpom_luck',
        'AdjEM': 'kenpom_adjem'
    }

    df.rename(col_mapping, axis=1, inplace=True)
    df = df[col_mapping.values()]

    # Clean up team names
    df.TeamName = df.TeamName.apply(
        lambda x: re.sub('\d', '', x).strip()).replace('.', '')

    add_team_id(df)

    df.to_csv(os.path.join(DATA_DIR, 'kenpom_data.csv'), index=False)

    return df


def add_team_id(df):
    '''
    Add team ID using team name and MTeams.csv file.
    '''

    df_spellings = pd.read_csv(os.path.join(DATA_DIR, "MTeamSpellings.csv"),
                               index_col=0, encoding='utf8')

    teams = df['TeamName'].apply(lambda x: x.lower().replace(';', ''))

    ids = teams.map(df_spellings['TeamID'])

    df['TeamID'] = ids


if __name__ == '__main__':
    years = base_urls.keys()
    df = get_kenpom_data(years)
